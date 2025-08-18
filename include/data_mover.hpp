#include "memory_manager.hpp"
#include "spdlog/spdlog.h"
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <logger.h>
#include <mutex>
#include <queue>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <torch/torch.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

struct FetchTask {
  void *dst_gpu_ptr;
  void *src_cpu_ptr;
  size_t size;
};

class DataMover {
public:
  static void init(Mode mode) {
    get_instance().mode_ = mode;
    inited = true;
  }

  static void *register_buffer(size_t size,
                               const std::string &shm_name = "haha") {
    assert(inited == true);
    if (get_instance().mode_ == Mode::GPU) {
      return get_instance().mem_manager_.register_gpu_buffer(size);
    } else {
      return get_instance().mem_manager_.register_shm_buffer(shm_name, size);
    }
  }

  static void load_file_to_buffer_sync(const std::string &shm_name,
                                       std::vector<std::string> weight_files) {
    LOG_ASSERT(inited == true, "Hasn't inited");
    auto &instance = get_instance();
    auto it = instance.mem_manager_.shm_buffers_.find(shm_name);
    if (it == instance.mem_manager_.shm_buffers_.end()) {
      spdlog::error("SHM buffer not found: {}", shm_name);
      return;
    }
    void *shm_ptr = it->second.addr;
    auto shm_size = it->second.size;
    // read from model files
    for (const auto &file_path : weight_files) {
      std::ifstream file(file_path, std::ios::binary | std::ios::ate);
      if (!file.is_open()) {
        spdlog::error("Failed to open file: {}", file_path);
        return;
      }
      size_t file_size = file.tellg();
      LOG_ASSERT(file_size <= shm_size, "Shm size: {}, file size: {}", shm_size,
                 file_size);
      file.seekg(0);

      file.read(static_cast<char *>(shm_ptr), file_size);
      file.close();
    }
  }

  static void load_buffer_to_gpu_sync(const std::string &src_shm_name,
                                      torch::Tensor tensor) {
    LOG_ASSERT(inited == true, "Hasn't inited");
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA");
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");

    auto &instance = get_instance();
    auto tensor_size = tensor.nbytes();
    if (instance.mode_ == Mode::CPU) {
      auto it = instance.mem_manager_.shm_buffers_.find(src_shm_name);
      if (it == instance.mem_manager_.shm_buffers_.end()) {
        std::cerr << "Cannot find " << src_shm_name << std::endl;
        return;
      }
      void *shm_ptr = it->second.addr;
      LOG_ASSERT(it->second.size - it->second.read_offset >= tensor_size,
                 "Tensor size {} > left size {}", tensor_size,
                 it->second.size - it->second.read_offset);
      cudaMemcpy(tensor.data_ptr(), shm_ptr, tensor_size,
                 cudaMemcpyHostToDevice);
      it->second.read_offset += tensor_size;
    }
  }

  static void print_shm_info(const std::string &shm_name) {
    auto &instance = get_instance();
    auto it = instance.mem_manager_.shm_buffers_.find(shm_name);
    if (it == instance.mem_manager_.shm_buffers_.end()) {
      std::cerr << "Cannot find " << shm_name << std::endl;
      return;
    }
    std::cout << it->second.size << " " << it->second.read_offset;
  }

  // static void

  // static void

  // static void fetch_data_async(void *dst_gpu_ptr, const std::string
  // &shm_name,
  //                              size_t size) {
  //   auto &instance = get_instance();
  //   auto it = instance.mem_manager_.shm_buffers_.find(shm_name);
  //   if (it == instance.mem_manager_.shm_buffers_.end()) {
  //     std::cerr << "SHM buffer not found: " << shm_name << std::endl;
  //     return;
  //   }
  //   void *src = it->second.addr;

  //   {
  //     std::lock_guard<std::mutex> lk(instance.queue_mutex_);
  //     instance.task_queue_.push({dst_gpu_ptr, src, size});
  //   }
  //   instance.cv_.notify_one();
  // }

  DataMover() {}

private:
  Mode mode_ = Mode::GPU;
  inline static std::atomic<bool> inited;
  MemoryManager mem_manager_;

  std::queue<FetchTask> task_queue_;
  std::mutex queue_mutex_;
  std::condition_variable cv_;
  std::thread worker_;
  bool stop_ = false;

  static DataMover &get_instance() {
    static DataMover instance;
    return instance;
  }
};
