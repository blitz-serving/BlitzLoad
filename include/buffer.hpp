

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
  Buffer(size_t buf_size, cudaStream_t *buf2tensor_stream, int device = 0)
      : buffer_size(buf_size) {
    cudaSetDevice(device);
    this->device = device;
    CUDA_CHECK(cudaMalloc(&buffer_ptr, buffer_size));
    this->buf2tensor_stream = *buf2tensor_stream;
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, buffer_ptr);
    spdlog::info("Buffer on device({})'s ptr on device({})", device,
                 attr.device);
  }

  Status mem_to_buffer(dangertensor::DangerTensor &source,
                       size_t &buffer_group_read_size) {
    if (status != EMPTY) {
      return status;
    }
    status = LOADING;
    auto [new_loaded_size, ended] =
        source.mem_to_buffer(buffer_ptr, buffer_size, buffer_group_read_size);
    this->local_usable_size += new_loaded_size;
    this->local_used_size = 0;
    buffer_group_read_size += new_loaded_size;
    cudaStreamSynchronize(0);
    status = READY;
    cv.notify_all();
    spdlog::info(
        "Buffer on device({}) new load bytes: 0x{:X}, cur usable size: 0x{:x} ",
        device, new_loaded_size, local_usable_size.load());
    return ended ? END // end: means dangertensor file has been all read
                 : READY;
  }

  std::pair<size_t, bool> export_handler(cudaIpcMemHandle_t *handle,
                                         size_t *offset, size_t tensor_size) {
    // LOG_ASSERT(status == READY || status == LOADED,
    //            "Hasn't loaded data, cannot load to tensor, "
    //            "current status: {}",
    //            to_string(status));
    std::unique_lock<std::mutex> guard(cv_mtx);
    cv.wait(guard,
            [this] { return this->status == READY || this->status == LOADED; });
    status = LOADED;
    auto load_tensor_size = tensor_size;
    if (tensor_size > buffer_size) {
      load_tensor_size = buffer_size;
    }
    LOG_ASSERT(local_usable_size >= planned_used_size + load_tensor_size,
               "usable size 0x{:x} < used 0x{:x} + tensor size 0x{:x}",
               local_usable_size.load(), planned_used_size.load(),
               load_tensor_size);
    __nv_bfloat16 *val =
        (__nv_bfloat16 *)((char *)buffer_ptr + planned_used_size);
    std::vector<__nv_bfloat16> vals(3);
    CUDA_CHECK(cudaMemcpy(vals.data(), (char *)buffer_ptr + planned_used_size,
                          sizeof(__nv_bfloat16) * 3, cudaMemcpyDeviceToHost));
    spdlog::info("Export Values: [{}, {}, {}]", __bfloat162float(vals[0]),
                 __bfloat162float(vals[1]), __bfloat162float(vals[2]));
    CUDA_CHECK(cudaIpcGetMemHandle(handle, buffer_ptr));
    *offset = planned_used_size.load();
    spdlog::info("Export tensor size: {}", load_tensor_size);
    planned_used_size += load_tensor_size;
    // spdlog::info("Export handler, length {}, planned:usable {}:{}",
    //              load_tensor_size, planned_used_size.load(),
    //              local_usable_size.load());
    if (planned_used_size == local_usable_size) {
      // cannot be written, only free handler can use this buffer now
      status = PLANNED_EMPTY;
      return {load_tensor_size, true};
    }
    return {load_tensor_size, false};
  }

  Status free_handler(size_t tensor_size) {
    local_used_size += tensor_size;
    if (local_used_size == local_usable_size) {
      // all tensors have been copied to python
      LOG_ASSERT(status == PLANNED_EMPTY,
                 "Buffer status should be planned_emtpy, not {}",
                 to_string(status));
      local_usable_size = 0;
      local_used_size = 0;
      planned_used_size = 0;
      status = EMPTY;
    }
    return status;
  }

  /// deprecated
  Status buffer_to_tensor(cudaIpcMemHandle_t &handle, size_t tensor_size,
                          int tensor_device) {
    cudaSetDevice(device);
    spdlog::info("Buffer to tensor, tensor device: {}", tensor_device);
    LOG_ASSERT(status == READY || status == LOADED,
               "Hasn't loaded data, cannot load to tensor on device {}, "
               "current status: {}",
               tensor_device, to_string(status));
    status = LOADED;
    void *tensor_ptr = nullptr;
    cudaIpcOpenMemHandle(&tensor_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, tensor_ptr);
    spdlog::info("Buffer get pointer on device({})", attr.device);
    if (tensor_device == device) {
      CUDA_CHECK(cudaMemcpyAsync(
          tensor_ptr, (char *)buffer_ptr + local_used_size, tensor_size,
          cudaMemcpyDeviceToDevice, buf2tensor_stream));
      CUDA_CHECK(cudaStreamSynchronize(buf2tensor_stream));
    } else {
      cudaMemcpyPeerAsync(tensor_ptr, tensor_device,
                          (char *)buffer_ptr + local_used_size, device,
                          tensor_size, buf2tensor_stream);
    }
    local_used_size += tensor_size;
    if (local_usable_size <= local_used_size) {
      LOG_ASSERT(
          local_usable_size == local_used_size,
          "usable size ({}) < used size ({}), there's some inconsistency",
          local_usable_size.load(), local_used_size.load());
      cudaStreamSynchronize(buf2tensor_stream);
      status = EMPTY;
    }
    spdlog::info("Buffer on device({}), used size: {}, status: {}", device,
                 local_used_size.load(), to_string(status.load()));
    return this->status.load();
  }

private:
  int device = -1;
  void *buffer_ptr;
  const size_t buffer_size;
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
      buffers.emplace_back(
          std::make_unique<Buffer>(single_buf_size, buf2tensor_stream, device));
      is_empty.emplace_back(1);
    }
  }

  Status mem_to_buffer(dangertensor::DangerTensor &source) {
    std::lock_guard<std::mutex> guard(mutexs[write_idx]);
    auto status =
        buffers[write_idx]->mem_to_buffer(source, buffer_group_read_size);
    if (status != EMPTY && status != END) {
      write_idx = (write_idx + 1) % group_size;
    }
    return status;
  }

  size_t export_handler(cudaIpcMemHandle_t *handle, size_t *offset,
                        size_t tensor_size) {
    std::lock_guard<std::mutex> guard(mutexs[read_idx]);
    auto [loaded_size, should_add_readidx] =
        buffers[read_idx]->export_handler(handle, offset, tensor_size);
    if (should_add_readidx) {
      read_idx = (read_idx + 1) % group_size;
    }
    return loaded_size;
  }

  void free_handler(size_t tensor_size) {
    std::lock_guard<std::mutex> guard(mutexs[read_idx]);
    auto status = buffers[release_idx]->free_handler(tensor_size);
    if (status == EMPTY) {
      release_idx = (release_idx + 1) % group_size;
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