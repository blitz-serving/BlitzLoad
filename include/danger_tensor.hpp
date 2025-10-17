#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cuda_bf16.h>
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

static const size_t ALIGN = 4096;

struct MetaData {
  uint64_t offset;
  uint64_t data_length;
  std::string name = "";
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
    if (valid) {
      return;
    }
    std::fstream f(danger_meta_path, std::ios::in);
    f >> tensor_num;
    metas_vec.resize(tensor_num);
    for (int i = 0; i < tensor_num; i++) {
      std::string tensor_name;
      uint64_t data_length;
      f >> tensor_name >> data_length;
      metas_vec[i] = {danger_tensor_file_size, data_length, tensor_name};
      danger_tensor_file_size += data_length;
    }
    spdlog::info("Load file {} done, tensor num: {}", danger_meta_path,
                 tensor_num);
  }

  /// mmap binfile to mock remote weight source
  void load_data_from_ssd(std::string danger_bin_path) {
    if (valid) {
      return;
    }
    file_read_mtx.lock();
    int fd = open(danger_bin_path.c_str(), O_DIRECT | O_RDONLY);
    if (fd == -1) {
      spdlog::error("Failed to open file: {}", danger_bin_path);
    }
    struct stat st;
    fstat(fd, &st);
    const size_t file_size = st.st_size;
    CUDA_CHECK(cudaMallocHost(&host_weight_segment, file_size));
    spdlog::info("Data file size: {}bytes, host weight segment: {:x}",
                 file_size, host_weight_segment);

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::future<int>> as;
    size_t total_chunks =
        (file_size + chunk_size_in_bytes - 1) / chunk_size_in_bytes;
    size_t chunk_per_thrd = (total_chunks + nthreads - 1) / nthreads;
    for (size_t i = 0; i < nthreads; ++i) {
      size_t partition_offset = i * chunk_per_thrd * chunk_size_in_bytes;
      if (i == nthreads - 1) {
        chunk_per_thrd = total_chunks - (nthreads - 2) * chunk_per_thrd;
      }

      as.emplace_back(
          std::async(std::launch::async, [this, file_size, fd, chunk_per_thrd,
                                          partition_offset]() {
            size_t offset = partition_offset;
            for (size_t i = 0; i < chunk_per_thrd; ++i) {
              size_t nbytes = std::min(chunk_size_in_bytes, file_size - offset);
              char *buf_ptr =
                  reinterpret_cast<char *>(this->host_weight_segment) + offset;
              ssize_t bytes_read =
                  pread_aligned(fd, buf_ptr, nbytes, offset, file_size);

              if (bytes_read < 0) {
                spdlog::error("Read chunk [0x{:x},0x{:x}] w/ errno: {} {}",
                              offset, offset + nbytes, errno, strerror(errno));
                return -1;
              } else if ((size_t)bytes_read != nbytes) {
                spdlog::error(
                    "Read chunk [{:x},{:x}] for {} bytes, but read {} bytes",
                    offset, offset + nbytes, nbytes, bytes_read);
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
    spdlog::info("Load model weight to host takes {} ms, effective bw {} GBps ",
                 duration.count(),
                 (double)file_size / duration.count() / 1000 / 1000);

    valid.store(true, std::memory_order_release);
    close(fd);
    file_read_mtx.unlock();
  }

  std::tuple<size_t, std::vector<size_t>, bool>
  mem_to_buffer(void *buffer_ptr, size_t buffer_size, size_t buffer_read_size,
                int buffer_idx, int device) {
    LOG_ASSERT(valid, "File hasn't been loaded");
    auto it = std::upper_bound(
        metas_vec.begin(), metas_vec.end(), buffer_read_size,
        [](size_t value, const MetaData &item) { return value < item.offset; });
    if (it != metas_vec.begin())
      it--;

    size_t loaded_size = 0, start_offset = buffer_read_size,
           first_tensor_offset = buffer_read_size - it->offset;
    std::vector<size_t> loaded_sizes;
    while (it != metas_vec.end() &&
           loaded_size + it->data_length - first_tensor_offset <= buffer_size) {
      auto ls = it->data_length - first_tensor_offset;
      __nv_bfloat16 *val =
          (__nv_bfloat16 *)(host_weight_segment + start_offset + loaded_size);
      loaded_size += ls;
      loaded_sizes.push_back(ls);
      first_tensor_offset = 0;
      spdlog::info(
          "[Buffer {}:{}] Loading {}, tensor_size 0x{:x}, cum_size 0x{:x}",
          device, buffer_idx, it->name, ls, loaded_size);
      // spdlog::info("[Buffer {}:{}] Loading {}, values: [{}, {}, {}]",
      // device,
      //              buffer_idx, it->name, __bfloat162float(*val),
      //              __bfloat162float(*(val + 1)), __bfloat162float(*(val +
      //              2)));
      it++;
    }
    if (it != metas_vec.end() && loaded_size == 0 &&
        it->data_length > buffer_size) {
      // single tensor size > buffer size, truncate
      __nv_bfloat16 *val =
          (__nv_bfloat16 *)(host_weight_segment + start_offset);
      spdlog::info("[Buffer {}:{}] loading partial {}, should truncate, "
                   "values: [{}, {}, {}]",
                   device, buffer_idx, it->name, __bfloat162float(*val),
                   __bfloat162float(*(val + 1)), __bfloat162float(*(val + 2)));
      loaded_size = buffer_size;
      loaded_sizes.push_back(buffer_size);
    }
    // spdlog::info("load size: 0x{:x}:0x{:x}, read done: {}", loaded_size,
    //              buffer_size, it == metas_vec.end());
    CUDA_CHECK(cudaMemcpyAsync(buffer_ptr, host_weight_segment + start_offset,
                               loaded_size, cudaMemcpyHostToDevice, 0));
    return {loaded_size, loaded_sizes, it == metas_vec.end()};
  }

private:
  ssize_t pread_aligned(int fd, void *buf, size_t nbytes, off_t offset,
                        size_t file_size) {
    if ((nbytes % ALIGN == 0) && (offset % ALIGN == 0) &&
        (((uintptr_t)buf) % ALIGN == 0)) {
      return pread(fd, buf, nbytes, offset);
    }

    size_t aligned_nbytes = ((nbytes + ALIGN - 1) / ALIGN) * ALIGN;

    void *tmp_buf = nullptr;
    if (posix_memalign(&tmp_buf, ALIGN, aligned_nbytes) != 0) {
      spdlog::error("posix_memalign failed");
      return -1;
    }

    ssize_t ret = pread(fd, tmp_buf, aligned_nbytes, offset);
    if (ret < 0) {
      spdlog::error("Read chunk [0x{:x},0x{:x}] w/ errno: {} {}", offset,
                    offset + nbytes, errno, strerror(errno));
      free(tmp_buf);
      return -1;
    }

    size_t copy_size = std::min((size_t)ret, nbytes);
    memcpy(buf, tmp_buf, copy_size);

    free(tmp_buf);
    return copy_size;
  }

private:
  int tensor_num;
  size_t danger_tensor_file_size = 0;
  // when buffer read from src, this will record read_offset and loaded_tensors
  std::atomic<int> loaded_tensors = 0;

  // std::atomic<size_t> meta_vec_idx = 0;
  std::vector<MetaData> metas_vec; // offset, length (SEQ)

  // used when there's no gdr, mock only
  std::mutex file_read_mtx;
  const int nthreads;
  const size_t chunk_size_in_bytes;
  char *host_weight_segment = nullptr;
  std::atomic<bool> valid = false;
};

} // namespace blitz::dangertensor