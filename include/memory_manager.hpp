#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <iostream>
#include <mutex>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

enum class Mode { GPU, CPU };

class MemoryManager {
public:
  ~MemoryManager() {
    std::cout << "destruct memory manager\n";
    for (auto &[ptr, size] : gpu_buffers_) {
      cudaFree(ptr);
    }
    for (auto &[name, shm_ptr] : shm_buffers_) {
      munmap(shm_ptr.addr, shm_ptr.size);
      shm_unlink(name.c_str());
    }
  }

  void *register_gpu_buffer(size_t size) {
    void *ptr = nullptr;
    cudaMalloc(&ptr, size);
    gpu_buffers_[ptr] = size;
    return ptr;
  }

  void *register_shm_buffer(const std::string &name, size_t size) {
    int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
    ftruncate(fd, size);
    void *addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    shm_buffers_[name] = {addr, size, 0};
    return addr;
  }

  struct ShmBuffer {
    void *addr;
    size_t size;
    size_t read_offset = 0;

    // ShmBuffer(void *a, size_t s) : addr(a), size(s), read_offset(0) {}
  };

  // private:
  std::unordered_map<void *, size_t> gpu_buffers_;
  std::unordered_map<std::string, ShmBuffer> shm_buffers_;
};
