#include "danger_tensor.hpp"
#include "logger.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <blitz_engine.h>
#include <common_tools.hpp>
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
}

size_t BlitzEngine::export_handler(cudaIpcMemHandle_t *handle, size_t *offset,
                                   size_t tensor_size, int rank) {
  return buf_groups[rank]->export_handler(handle, offset, tensor_size);
}

void BlitzEngine::free_handler(size_t tensor_size, int rank) {
  buf_groups[rank]->free_handler(tensor_size);
}

void BlitzEngine::mem_to_buffer(std::string model_path, int rank_num) {
  spdlog::info("Trigger mem2buf");
  for (int _rank = 0; _rank < rank_num; _rank++) {
    threads.emplace_back(std::thread([this, _rank, model_path]() {
      while (true) {
        LOG_ASSERT(this->buf_groups[_rank] != nullptr,
                   "BufGroup is null, cur rank: {}, group size: {}", _rank,
                   this->buf_groups.size());
        auto status = this->buf_groups[_rank]->mem_to_buffer(
            *(dangertensor_map[model_path][_rank]));
        if (status == buffer::END) {
          spdlog::info("Buffers[{}] load done", _rank);
          break;
        }
        std::this_thread::yield();
      }
    }));
  }
}

int BlitzEngine::pull_model(std::string model_name_or_path) {
  spdlog::info("Want to load model: {}", model_name_or_path);
  std::regex pattern(R"(dangertensors\.(\d+)\.bin)");
  int rank_num = 0;
  try {
    for (const auto &entry : fs::directory_iterator(model_name_or_path)) {
      if (entry.is_regular_file()) {
        std::string filename = entry.path().filename().string();
        if (std::regex_match(filename, pattern)) {
          rank_num++;
          // find rank
          std::smatch match;
          std::regex_match(filename, match, pattern);
          int rank = std::stoi(match[1].str());
          spdlog::info("Match file {}, rank is {}", filename, rank);
          dangertensor_map[model_name_or_path][rank] =
              std::make_unique<dangertensor::DangerTensor>();

          // load files
          auto bin_file = entry.path().string();
          auto meta_file = bin_file;
          auto pos = meta_file.rfind('.');
          meta_file.replace(pos + 1, meta_file.size() - pos - 1, "meta");
          auto danger_tensor = dangertensor_map[model_name_or_path][rank].get();
          danger_tensor->load_meta_from_ssd(meta_file);
          danger_tensor->load_data_from_ssd(bin_file);
        }
      }
    }
  } catch (const std::exception &e) {
    spdlog::error("Error: {}", e.what());
  }
  spdlog::info("Load done");
  return rank_num;
}

BlitzEngine::~BlitzEngine() {
  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}