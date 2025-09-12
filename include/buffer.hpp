

#include "danger_tensor.hpp"
#include "spdlog/spdlog.h"
#include <atomic>
#include <cstddef>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <logger.h>
#include <memory>
#include <mutex>
#include <spin_mutex.hpp>
#include <vector>

namespace blitz::buffer {

/// emtpy -> loading -> ready -> loaded -> emtpy -> ... -> end
enum Status {
  EMPTY,
  LOADING,
  READY,
  LOADED,
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
    // std::cout << "Pointer is on device " << attr.device << std::endl;
  }

  Status mem_to_buffer(dangertensor::DangerTensor &source) {
    LOG_ASSERT(false, "cannot use this func");
    return EMPTY;
    // LOG_ASSERT(status == EMPTY,
    //  "Data hasn't been loaded, cannot load new data");
    // if (status == READY || status == END || status == LOADED) {
    //   return status;
    // }
    // status = LOADING;
    // auto new_loaded_size = source.mem_to_buffer(buffer_ptr, buffer_size);
    // this->local_usable_size += new_loaded_size;
    // this->local_used_size = 0;
    // cudaStreamSynchronize(0);
    // status = READY;
    // spdlog::info("Buffer on device({}) new load bytes: {}", device,
    //              new_loaded_size);
    // return this->local_usable_size > 0 ? READY : END; // end: means dangertensor
    //                                                   // file has been all read
  }

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
      cudaMemcpyAsync(tensor_ptr, (char *)buffer_ptr + local_used_size,
                      tensor_size, cudaMemcpyDeviceToDevice, buf2tensor_stream);
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
                 local_used_size.load(), to_string(status));
    return this->status;
  }

private:
  int device = -1;
  void *buffer_ptr;
  const size_t buffer_size;
  Status status = EMPTY;
  std::atomic<size_t> local_usable_size = 0, local_used_size = 0;
  cudaStream_t buf2tensor_stream;
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
    for (int i = 0; i < group_size; i++) {
      buffers.emplace_back(
          std::make_unique<Buffer>(single_buf_size, buf2tensor_stream, device));
    }
  }

  Status mem_to_buffer(dangertensor::DangerTensor &source) {
    std::lock_guard<std::mutex> guard(mutexs[write_idx]);
    auto status = buffers[write_idx]->mem_to_buffer(source);
    if (status != EMPTY && status != END) {
      // empty: hasn't read to tensor, end: all weights has been written to
      // buffer
      write_idx = (write_idx + 1) % group_size;
      spdlog::info("Buffer full, write idx ++");
    }
    return status;
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
  const int group_size, device;
  std::atomic<int> read_idx;
  std::atomic<int> write_idx;
};

} // namespace blitz::buffer