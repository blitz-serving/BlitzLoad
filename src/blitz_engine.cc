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
#include <map>
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
}

std::pair<size_t, bool> BlitzEngine::export_handler(cudaIpcMemHandle_t *handle,
                                                    size_t *offset,
                                                    size_t tensor_size,
                                                    int rank) {
  return buf_groups[rank]->export_handler(handle, offset, tensor_size);
}

void BlitzEngine::free_handler(size_t tensor_size, int rank) {
  buf_groups[rank]->free_handler(tensor_size);
}

void BlitzEngine::mem_to_buffer(std::string danger_tensor_index_name,
                                int rank_num) {
  spdlog::info("Trigger mem2buf, dangertensor index {}, ranks {}",
               danger_tensor_index_name, rank_num);
  for (int _rank = 0; _rank < rank_num; _rank++) {
    threads.emplace_back(std::thread([this, _rank, danger_tensor_index_name]() {
      while (true) {
        auto status = this->buf_groups[_rank]->mem_to_buffer(
            *(dangertensor_map[danger_tensor_index_name][_rank]));
        if (status == buffer::END) {
          spdlog::info("Buffers[{}] load done", _rank);
          break;
        }
        std::this_thread::yield();
      }
    }));
  }
}

void BlitzEngine::reset_status(int rank) {
  spdlog::info("Reset buffers status on rank {}", rank);
  this->buf_groups[rank]->reset_status();
}

int BlitzEngine::pull_model(std::string model_name_or_path, int tp_size,
                            int pp_size) {
  spdlog::info("Want to load model: {}", model_name_or_path);
  std::regex pattern(R"(dangertensors\.(\d+)\.bin)");
  int rank_num = 0;
  auto rank_file_map = std::map<int, std::string>();
  auto danger_tensor_index_name =
      gen_dangertensor_index_name(model_name_or_path, tp_size, pp_size);
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
          rank_file_map[rank] = entry.path().string();
          if (dangertensor_map.find(danger_tensor_index_name) ==
                  dangertensor_map.end() ||
              dangertensor_map[danger_tensor_index_name].find(rank) ==
                  dangertensor_map[danger_tensor_index_name].end()) {
            // cannot find danger_tensor
            spdlog::info("Create dangertensor {}, rank {}",
                         danger_tensor_index_name, rank);
            dangertensor_map[danger_tensor_index_name][rank] =
                std::make_unique<dangertensor::DangerTensor>();
          } else {
            spdlog::debug("DangerTensor {}:{} existed",
                          danger_tensor_index_name, rank);
          }
        }
      }
    }
    for (auto [rank, entry_name] : rank_file_map) {
      load_file_to_mem(entry_name, rank, danger_tensor_index_name);
    }
  } catch (const std::exception &e) {
    spdlog::error("Error: {}", e.what());
  }
  spdlog::info("Load done");
  return rank_num;
}

void BlitzEngine::load_file_to_mem(std::string file, int rank,
                                   std::string dangertensor_index) {
  auto bin_file = file;
  auto meta_file = bin_file;
  auto pos = meta_file.rfind('.');
  meta_file.replace(pos + 1, meta_file.size() - pos - 1, "meta");
  auto danger_tensor = dangertensor_map[dangertensor_index][rank].get();
  if (danger_tensor == nullptr) {
    dangertensor_map[dangertensor_index][rank] =
        std::make_unique<dangertensor::DangerTensor>();
  }
  danger_tensor->load_meta_from_ssd(meta_file);
  danger_tensor->load_data_from_ssd(bin_file);
  spdlog::info("{} load done", bin_file);
}

BlitzEngine::~BlitzEngine() {
  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}