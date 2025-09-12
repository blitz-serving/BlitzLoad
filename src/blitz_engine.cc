#include "danger_tensor.hpp"
#include "logger.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <blitz_engine.h>
#include <cstddef>
#include <filesystem>
#include <iterator>
#include <memory>
#include <regex>
#include <string>
#include <thread>
#include <vector>

using namespace blitz;
namespace fs = std::filesystem;

BlitzEngine::BlitzEngine(std::vector<int> buf_devices_, size_t buf_size) {
  std::sort(buf_devices_.begin(), buf_devices_.end());
  this->buf_devices = buf_devices_;
  // create buf2tensor stream
  CUDA_CHECK(cudaStreamCreate(&this->buf2tensor_stream));
  auto device_iter = buf_devices_.begin();

  while (device_iter != buf_devices_.end()) {
    bufs.emplace_back(std::make_unique<buffer::BufferGroup>(
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

void BlitzEngine::mem_to_tensor(cudaIpcMemHandle_t &handle,
                                std::string tensor_name, size_t tensor_size,
                                int tensor_device) {
  // FIXME: hardcode
  LOG_ASSERT(dangertensor_map.find(0) != dangertensor_map.end(),
             "Hasn't inited dangertensor[0]");
  dangertensor_map[0]->mem_to_tensor(handle, tensor_name, tensor_size,
                                     tensor_device);
}

void BlitzEngine::buffer_to_tensor(cudaIpcMemHandle_t &handle,
                                   int tensor_device, size_t tensor_size) {
  LOG_ASSERT(false, "Cannot use this func now");
  auto iter = std::find(buf_devices.begin(), buf_devices.end(), tensor_device);
  int idx = 0;
  if (iter != buf_devices.end()) {
    idx = std::distance(iter, buf_devices.begin());
  }
  auto status = bufs[idx]->buffer_to_tensor(handle, tensor_size, tensor_device);
}

void BlitzEngine::mem_to_buffer(std::vector<std::string> files) {
  LOG_ASSERT(false, "Cannot use this func now");
  auto idx = 0;
  for (auto &file : files) {
    threads.emplace_back(std::thread([this, file, idx]() {
      int shard_id = -1;
      std::regex pattern(R"(dangertensors\.(\d+)\.bin)");
      std::smatch match;
      if (std::regex_match(file, match, pattern)) {
        shard_id = std::stoi(match[1].str());
      }
      while (true) {
        auto status =
            this->bufs[idx]->mem_to_buffer(*dangertensor_map[shard_id]);
        if (status == buffer::END) {
          spdlog::info("Buffers[{}] load done", idx);
          break;
        }
        std::this_thread::yield();
      }
    }));
    idx += 1;
  }
}

void BlitzEngine::ssd_to_mem(std::vector<std::string> bin_files) {
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
    if (std::regex_match(filename, match, pattern)) {
      int shard_id = std::stoi(match[1].str());
      spdlog::info("Match file {}, meta file is {}, shard_id is {}", bin_file,
                   meta_file, shard_id);
      dangertensor_map[shard_id] =
          std::make_unique<dangertensor::DangerTensor>();
      auto danger_tensor = dangertensor_map[shard_id].get();
      danger_tensor->load_meta_from_ssd(meta_file);
      danger_tensor->load_data_from_ssd(bin_file);
    }
  }
  spdlog::info("Load done");
  spdlog::default_logger()->flush();
}

BlitzEngine::~BlitzEngine() {
  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}