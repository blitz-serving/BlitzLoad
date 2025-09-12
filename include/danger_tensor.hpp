#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <spdlog/spdlog.h>
#include <vector>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <logger.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace blitz::dangertensor {

struct MetaData {
  uint64_t offset;
  uint64_t data_length;
  std::string name = "";
};

enum ReadMode {
  NAME_MATCH,
  SEQ,
};

/// One dangertensor file pair -> one DangerTensor object.
///
/// If TP=2, there will be two objects.
class DangerTensor {
public:
  DangerTensor(size_t chunk_size = 512 * 1024 * 1024, int thread_num = 5)
      : chunk_size_in_bytes(chunk_size), nthreads(thread_num) {}

  /// read meta file to mock remote transfer
  void load_meta_from_ssd(std::string danger_meta_path) {
    std::fstream f(danger_meta_path, std::ios::in);
    f >> tensor_num;
    metas_vec.resize(tensor_num);
    for (int i = 0; i < tensor_num; i++) {
      std::string tensor_name;
      uint64_t data_length;
      f >> tensor_name >> data_length;
      metas_map[tensor_name] = {danger_tensor_file_size, data_length};
      metas_vec[i] = {danger_tensor_file_size, data_length, tensor_name};
      danger_tensor_file_size += data_length;
    }
    spdlog::info("Load file {} done, tensor num: {}", danger_meta_path,
                 tensor_num);
  }

  /// mmap binfile to mock remote weight source
  void load_data_from_ssd(std::string danger_bin_path) {
    file_read_mtx.lock();
    int fd = open(danger_bin_path.c_str(), O_DIRECT | O_RDONLY);
    if (fd == -1) {
      spdlog::error("Failed to open file: {}", danger_bin_path);
    }
    struct stat st;
    fstat(fd, &st);
    const size_t file_size = st.st_size;
    CUDA_CHECK(cudaMallocHost(&host_weight_segment, file_size));
    spdlog::info("Data file size: {}bytes, host weight segment: {:#x}",
                 file_size, host_weight_segment);

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::future<int>> as;
    for (size_t i = 0; i < nthreads; ++i) {
      size_t total_chunks =
          (file_size + chunk_size_in_bytes - 1) / chunk_size_in_bytes;
      size_t chunk_per_thrd = (i < nthreads - 1)
                                  ? (total_chunks + nthreads - 1) / nthreads
                                  : total_chunks / nthreads;
      size_t partition_offset = i * chunk_per_thrd * chunk_size_in_bytes;

      as.emplace_back(
          std::async(std::launch::async, [this, file_size, fd, chunk_per_thrd,
                                          partition_offset]() {
            size_t offset = partition_offset;
            for (size_t i = 0; i < chunk_per_thrd; ++i) {
              size_t nbytes = std::min(chunk_size_in_bytes, file_size - offset);
              char *buf_ptr =
                  reinterpret_cast<char *>(this->host_weight_segment) + offset;
              ssize_t bytes_read = pread(fd, buf_ptr, nbytes, offset);

              if (bytes_read < 0) {
                spdlog::error("Read chunk [{:#x},{:#x}] w/ errno: {} {}",
                              offset, offset + nbytes, errno, strerror(errno));
                return -1;
              } else if ((size_t)bytes_read != nbytes) {
                spdlog::error(
                    "Read chunk [{:#x},{:#x}] for {} bytes, but read {} bytes",
                    offset, offset + nbytes, bytes_read);
                return -2;
              }

              offset += bytes_read;
            }
            return 0;
          }));
    }

    for (auto &a : as) {
      a.wait();
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    // spdlog::info("Load model weight to host takes {} ms, effective bw {}
    // GBps",
    //              duration, (double)file_size / duration.count() / 1000 /
    //              1000);

    valid.store(true, std::memory_order_release);
    close(fd);
    file_read_mtx.unlock();
  }

  void mem_to_tensor(cudaIpcMemHandle_t &handle, std::string tensor_name,
                     size_t tensor_length, int tensor_device) {
    LOG_ASSERT(valid, "File hasn't been loaded");

    cudaSetDevice(tensor_device);
    void *tensor_ptr = nullptr;
    cudaIpcOpenMemHandle(&tensor_ptr, handle, cudaIpcMemLazyEnablePeerAccess);

    if (!tensor_name.empty()) {
      LOG_ASSERT(mode != SEQ,
                 "Has inited as sequential mode, cannot find by name");
      LOG_ASSERT(metas_map.find(tensor_name) != metas_map.end(),
                 "Cannot find tensor {}", tensor_name);
      auto [offset, length, _name] = metas_map[tensor_name];
      LOG_ASSERT(length == tensor_length,
                 "Input tensor_length {} != metas's tensor_length {}",
                 tensor_length, length);

      spdlog::info("Loading tensor {}, length {}", tensor_name, tensor_length);
      cudaMemcpyAsync(tensor_ptr, host_weight_segment + offset, tensor_length,
                      cudaMemcpyHostToDevice, 0);
      CUDA_CHECK(cudaStreamSynchronize(0));
    } else {
      // sequential mode
      mode = SEQ;
      auto idx = meta_vec_idx.fetch_add(1);
      auto [offset, length, name] = metas_vec[idx];
      LOG_ASSERT(length == tensor_length,
                 "Input tensor_length {} != metas's tensor {}, length {}",
                 tensor_length, name, length);
      spdlog::info("Loading tensor {}, length {}", name, tensor_length);
      cudaMemcpyAsync(tensor_ptr, host_weight_segment + offset, tensor_length,
                      cudaMemcpyHostToDevice, 0);
      CUDA_CHECK(cudaStreamSynchronize(0));
    }
  }

private:
  int tensor_num;
  size_t danger_tensor_file_size = 0;
  // when buffer read from src, this will record read_offset and loaded_tensors
  std::atomic<int> loaded_tensors = 0;

  ReadMode mode = NAME_MATCH;
  std::atomic<size_t> meta_vec_idx = 0;
  std::map<std::string, MetaData>
      metas_map;                   // name, offset, length (NAME_MATCH)
  std::vector<MetaData> metas_vec; // offset, length (SEQ)

  std::mutex file_read_mtx;
  const int nthreads;
  const size_t chunk_size_in_bytes;
  char *host_weight_segment = nullptr;
  std::atomic<bool> valid = false;

  // used when there's no gdr, mock only
  void *weight_file_ptr;
};

} // namespace blitz::dangertensor