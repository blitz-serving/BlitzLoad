

#include "danger_tensor.hpp"
#include "spdlog/spdlog.h"
#include <atomic>
#include <cstddef>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <logger.h>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

namespace blitz::buffer {

/// emtpy -> loading -> ready -> loaded -> emtpy -> ... -> end
enum Status {
  EMPTY,
  LOADING,
  READY,
  LOADED,
  PLANNED_EMPTY,
  END,
};

inline std::string to_string(blitz::buffer::Status s) {
  switch (s) {
  case blitz::buffer::Status::EMPTY:
    return "EMTPY";
  case blitz::buffer::Status::LOADING:
    return "LOADING";
  case blitz::buffer::Status::READY:
    return "READY";
  case blitz::buffer::Status::LOADED:
    return "LOADED";
  case blitz::buffer::Status::END:
    return "END";
  default:
    return "Unknown";
  }
}

// HBM buffer
class Buffer {
public:
  Buffer(size_t buf_size, cudaStream_t *buf2tensor_stream, int idx,
         int device = 0)
      : buffer_size(buf_size), buffer_idx(idx) {
    cudaSetDevice(device);
    this->device = device;
    CUDA_CHECK(cudaMalloc(&buffer_ptr, buffer_size));
    this->buf2tensor_stream = *buf2tensor_stream;
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, buffer_ptr);
    spdlog::info("[Buffer {}:{}] Init", device, buffer_idx);
  }

  Status mem_to_buffer(dangertensor::DangerTensor &source,
                       size_t &buffer_group_read_size) {
    if (status != EMPTY) {
      return status;
    }
    status = LOADING;
    auto [new_loaded_size, loaded_sizes_, ended] = source.mem_to_buffer(
        buffer_ptr, buffer_size, buffer_group_read_size, buffer_idx, device);
    this->local_usable_size += new_loaded_size;
    this->local_used_size = 0;
    this->loaded_sizes = loaded_sizes_;
    buffer_group_read_size += new_loaded_size;
    cudaStreamSynchronize(0);
    status = READY;
    cv.notify_all();
    spdlog::info(
        "[Buffer {}:{}] new load bytes: 0x{:X}, cur usable size: 0x{:x} ",
        device, buffer_idx, new_loaded_size, local_usable_size.load());
    return ended ? END // end: means dangertensor file has been all read
                 : READY;
  }

  std::tuple<size_t, bool, bool> export_handler(cudaIpcMemHandle_t *handle,
                                                size_t *offset,
                                                size_t tensor_size) {
    std::unique_lock<std::mutex> guard(cv_mtx);
    cv.wait(guard,
            [this] { return this->status == READY || this->status == LOADED; });
    status = LOADED;
    bool tensor_size_too_large = false;
    auto load_tensor_size = tensor_size;
    if (tensor_size > buffer_size) {
      load_tensor_size = buffer_size;
    }
    auto should_load_size = std::move(loaded_sizes.front());
    loaded_sizes.erase(loaded_sizes.begin());

    if (should_load_size < load_tensor_size) {
      spdlog::debug("Tensor size is larger {} > {}", load_tensor_size,
                    should_load_size);
      load_tensor_size = should_load_size;
      tensor_size_too_large = true;
    }

    LOG_ASSERT(
        local_usable_size >= planned_used_size + load_tensor_size,
        "[Buffer {}:{}] usable size 0x{:x} < used 0x{:x} + tensor size 0x{:x}",
        device, buffer_idx, local_usable_size.load(), planned_used_size.load(),
        load_tensor_size);
    __nv_bfloat16 *val =
        (__nv_bfloat16 *)((char *)buffer_ptr + planned_used_size);
    std::vector<__nv_bfloat16> vals(3);
    CUDA_CHECK(cudaMemcpy(vals.data(), (char *)buffer_ptr + planned_used_size,
                          sizeof(__nv_bfloat16) * 3, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaIpcGetMemHandle(handle, buffer_ptr));
    *offset = planned_used_size.load();
    planned_used_size += load_tensor_size;
    spdlog::info("[Buffer {}:{}] export cum size: {:x}, tensor size: {:x}",
                 device, buffer_idx, planned_used_size.load(),
                 load_tensor_size);
    if (planned_used_size == local_usable_size) {
      // cannot be written, only free handler can use this buffer now
      status = PLANNED_EMPTY;
      return {load_tensor_size, true, tensor_size_too_large};
    }
    return {load_tensor_size, false, tensor_size_too_large};
  }

  Status free_handler(size_t tensor_size) {
    local_used_size += tensor_size;
    if (local_used_size == local_usable_size) {
      // all tensors have been copied to python
      LOG_ASSERT(status == PLANNED_EMPTY,
                 "[Buffer {}:{}] status should be planned_emtpy, not {}",
                 device, buffer_idx, to_string(status));
      local_usable_size = 0;
      local_used_size = 0;
      planned_used_size = 0;
      status = EMPTY;
    }
    return status;
  }

  void reset_status() {
    local_usable_size = 0;
    local_used_size = 0;
    planned_used_size = 0;
    status = EMPTY;
  }

  /// deprecated
  Status buffer_to_tensor(cudaIpcMemHandle_t &handle, size_t tensor_size,
                          int tensor_device) {
    LOG_ASSERT(false, "Deprecated method");
    return this->status.load();
  }

private:
  int device = -1;
  int buffer_idx = 0;
  void *buffer_ptr;
  const size_t buffer_size;
  std::vector<size_t> loaded_sizes;
  std::atomic<Status> status = EMPTY;
  std::atomic<size_t> local_usable_size = 0, local_used_size = 0,
                      planned_used_size = 0;
  cudaStream_t buf2tensor_stream;
  std::condition_variable cv;
  std::mutex cv_mtx;
};

/// HBM buffer group
class BufferGroup {
public:
  BufferGroup(int group_size, size_t buf_size, cudaStream_t *buf2tensor_stream,
              int device = 0)
      : group_size(group_size), device(device) {
    auto single_buf_size = buf_size / group_size;
    mutexs = std::vector<std::mutex>(group_size);
    read_idx = 0;
    write_idx = 0;
    release_idx = 0;
    buffer_group_read_size = 0;
    for (int i = 0; i < group_size; i++) {
      buffers.emplace_back(std::make_unique<Buffer>(
          single_buf_size, buf2tensor_stream, i, device));
      is_empty.emplace_back(1);
    }
  }

  Status mem_to_buffer(dangertensor::DangerTensor &source) {
    std::lock_guard<std::mutex> guard(mutexs[write_idx]);
    auto status =
        buffers[write_idx]->mem_to_buffer(source, buffer_group_read_size);
    if (status == READY) {
      write_idx = (write_idx + 1) % group_size;
    }
    // if (status != READY && status != END) {
    //   write_idx = (write_idx + 1) % group_size;
    // }
    return status;
  }

  std::pair<size_t, bool> export_handler(cudaIpcMemHandle_t *handle,
                                         size_t *offset, size_t tensor_size) {
    std::lock_guard<std::mutex> guard(mutexs[read_idx]);
    auto [loaded_size, should_add_readidx, tensor_size_too_large] =
        buffers[read_idx]->export_handler(handle, offset, tensor_size);
    if (should_add_readidx) {
      read_idx = (read_idx + 1) % group_size;
    }
    return {loaded_size, tensor_size_too_large};
  }

  void free_handler(size_t tensor_size) {
    std::lock_guard<std::mutex> guard(mutexs[release_idx]);
    auto status = buffers[release_idx]->free_handler(tensor_size);
    if (status == EMPTY) {
      release_idx = (release_idx + 1) % group_size;
    }
  }

  void reset_status() {
    for (int i = 0; i < group_size; i++) {
      mutexs[i].lock();
    }
    read_idx = 0;
    write_idx = 0;
    release_idx = 0;
    buffer_group_read_size = 0;
    for (std::unique_ptr<Buffer> &buf_ptr : buffers) {
      buf_ptr->reset_status();
    }
    for (int i = group_size - 1; i >= 0; i--) {
      mutexs[i].unlock();
    }
  }

  Status buffer_to_tensor(cudaIpcMemHandle_t &handle, size_t tensor_size,
                          int tensor_device) {
    std::lock_guard<std::mutex> guard(mutexs[read_idx]);
    auto status =
        buffers[read_idx]->buffer_to_tensor(handle, tensor_size, tensor_device);
    if (status == EMPTY) {
      read_idx = (read_idx + 1) % group_size;
    }
    return status;
  }

private:
  std::vector<std::unique_ptr<Buffer>> buffers;
  std::vector<std::mutex> mutexs;
  std::vector<int> is_empty;
  const int group_size, device;
  std::atomic<int> read_idx;
  std::atomic<int> release_idx;
  std::atomic<int> write_idx;
  size_t buffer_group_read_size;
};

} // namespace blitz::buffer