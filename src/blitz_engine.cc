#include "danger_tensor.hpp"
#include "logger.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <blitz_engine.h>
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <iterator>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

using namespace blitz;
namespace fs = std::filesystem;

BlitzEngine::BlitzEngine(std::vector<int> buf_devices_, size_t buf_size) {
  std::sort(buf_devices_.begin(), buf_devices_.end());
  this->buf_devices = buf_devices_;
  spdlog::info("Buffer size: {}, {}", buf_devices_.size(),
               this->buf_devices.size());
  // create buf2tensor stream
  CUDA_CHECK(cudaStreamCreate(&this->buf2tensor_stream));
  auto device_iter = buf_devices_.begin();

  while (device_iter != buf_devices_.end()) {
    buf_groups.emplace_back(std::make_unique<buffer::BufferGroup>(
        2, buf_size, &buf2tensor_stream, *device_iter));
    device_iter++;
  }
  this->enable_p2p_access(buf_devices_);
  for (size_t i = 0; i < buf_devices_.size(); i++) {
    is_empty.push_back(true);
    mtxs.push_back(std::make_unique<std::mutex>());
    cvs.push_back(std::make_unique<std::condition_variable>());
  }
  spdlog::info("length: mtx: {}, cv: {}, is_empty: {}", mtxs.size(), cvs.size(),
               is_empty.size());
  auto test = cvs[0].get();
  test->notify_one();
}

void BlitzEngine::enable_p2p_access(std::vector<int> devices) {
  auto device_size = devices.size();
  int can_access = false;
  for (size_t i = 0; i < device_size; i++) {
    cudaSetDevice(devices[i]);
    for (size_t j = i + 1; j < device_size; j++) {
      CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, devices[i], devices[j]));
      if (can_access) {
        CUDA_CHECK(cudaDeviceEnablePeerAccess(j, 0));
      }
    }
  }
}

/// deprecated
void BlitzEngine::mem_to_tensor(cudaIpcMemHandle_t &handle,
                                std::string tensor_name, size_t tensor_size,
                                int tensor_device) {
  LOG_ASSERT(false, "deprecated method");
  // LOG_ASSERT(dangertensor_map.find(0) != dangertensor_map.end(),
  //            "Hasn't inited dangertensor[0]");
  // dangertensor_map[0]->mem_to_tensor(handle, tensor_name, tensor_size,
  //                                    tensor_device);
}

void BlitzEngine::buffer_to_tensor(cudaIpcMemHandle_t &handle,
                                   int tensor_device, size_t tensor_size,
                                   int rank) {
  LOG_ASSERT(std::find(buf_devices.begin(), buf_devices.end(), tensor_device) !=
                 buf_devices.end(),
             "Hasn't access p2p");
  auto status =
      buf_groups[rank]->buffer_to_tensor(handle, tensor_size, tensor_device);
  if (status == buffer::EMPTY) {
    is_empty[rank] = 1;
    cvs[rank]->notify_one();
  }
}

size_t BlitzEngine::export_handler(cudaIpcMemHandle_t *handle, size_t *offset,
                                   size_t tensor_size, int rank) {
  return buf_groups[rank]->export_handler(handle, offset, tensor_size);
}

void BlitzEngine::free_handler(size_t tensor_size, int rank) {
  buf_groups[rank]->free_handler(tensor_size);
}

void BlitzEngine::mem_to_buffer(std::string model_path, int rank_num) {
  for (int _rank = 0; _rank < rank_num; _rank++) {
    threads.emplace_back(std::thread([this, _rank, model_path]() {
      while (true) {
        auto status = this->buf_groups[_rank]->mem_to_buffer(
            *(dangertensor_map[model_path][_rank]));
        if (status == buffer::END) {
          spdlog::info("Buffers[{}] load done", _rank);
          break;
        }
        if (status == buffer::READY || status == buffer::END ||
            status == buffer::LOADED) {
          is_empty[_rank] = 0;
        }
        std::this_thread::yield();
      }
    }));
  }
}

std::pair<std::string, int>
BlitzEngine::ssd_to_mem(std::vector<std::string> bin_files) {
  std::string model_path = "";
  for (auto bin_file : bin_files) {
    auto meta_file = bin_file;
    size_t pos = meta_file.rfind('.');
    if (pos != std::string::npos) {
      meta_file.replace(pos + 1, meta_file.size() - pos - 1, "meta");
    }
    spdlog::info("Current bin_file: {}, meta_file: {}", bin_file, meta_file);
    std::regex pattern(R"(dangertensors\.(\d+)\.bin)");
    std::smatch match;
    fs::path p(bin_file);
    auto filename = p.filename().string();
    model_path = p.parent_path().string();
    if (std::regex_match(filename, match, pattern)) {
      int rank = std::stoi(match[1].str());
      spdlog::info("Match file {}, meta file is {}, rank is {}", bin_file,
                   meta_file, rank);
      dangertensor_map[model_path][rank] =
          std::make_unique<dangertensor::DangerTensor>();
      auto danger_tensor = dangertensor_map[model_path][rank].get();
      danger_tensor->load_meta_from_ssd(meta_file);
      danger_tensor->load_data_from_ssd(bin_file);
    }
  }
  spdlog::info("Load done");
  return {model_path, bin_files.size()};
}

BlitzEngine::~BlitzEngine() {
  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}