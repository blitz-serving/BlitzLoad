#pragma once
#include "danger_tensor.hpp"
#include <array>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace std;
using nlohmann::json;

struct PullModelRequest {
  string model_name;
  int32_t world_size;
  int32_t tp_size;
  int32_t pp_size;
};

struct PullModelResponse {
  string task_id;
};

struct PullDiffusionModelRequest {
  string file_name;
};

struct CheckModelRequest {
  string model_name;
  string task_id;
};

struct CheckModelResponse {
  bool done;
};

struct LoadTensorRequest {
  string tensor_name;
  uint64_t tensor_size;
  int32_t rank;
};

struct LoadTensorResponse {
  cudaIpcMemHandle_t handler;
  uint64_t offset;
  uint64_t loaded_size;
  bool resize_tensor;
};

struct RevertHandlerRequest {
  string tensor_name;
  uint64_t tensor_size;
  int32_t rank;
};

struct RevertHandlerResponse {
  bool success;
};

struct GetMetaRequest {
  string file_name;
};

struct GetMetaResponse {
  string meta_str;
};

struct GetMetaTensorRequest {
  string file_name;
};

struct GetMetaTensorResponse {
  std::vector<blitz::dangertensor::MetaData> meta_tensors;
};

struct ResetStatusRequest {
  int rank;
};

struct EmptyRequestResponse {};

// ------------------- JSON -------------------

inline void from_json(const json &j, GetMetaRequest &r) {
  j.at("file_name").get_to(r.file_name);
}

inline void to_json(json &j, const GetMetaRequest &r) {
  j = json{{"file_name", r.file_name}};
}

inline void from_json(const json &j, GetMetaResponse &r) {
  j.at("meta_str").get_to(r.meta_str);
}

inline void to_json(json &j, const GetMetaResponse &r) {
  j = json{{"meta_str", r.meta_str}};
}

inline void from_json(const json &j, GetMetaTensorRequest &r) {
  j.at("file_name").get_to(r.file_name);
}

inline void to_json(json &j, const GetMetaTensorRequest &r) {
  j = json{{"file_name", r.file_name}};
}

inline void from_json(const json &j, GetMetaTensorResponse &r) {
  j.at("meta_tensors").get_to(r.meta_tensors);
}

inline void to_json(json &j, const GetMetaTensorResponse &r) {
  j = json{{"meta_tensors", r.meta_tensors}};
}

inline void from_json(const json &j, PullModelRequest &r) {
  j.at("model_name").get_to(r.model_name);
  j.at("world_size").get_to(r.world_size);
  j.at("tp_size").get_to(r.tp_size);
  j.at("pp_size").get_to(r.pp_size);
}

inline void to_json(json &j, const PullModelRequest &r) {
  j = json{{"model_name", r.model_name},
           {"world_size", r.world_size},
           {"tp_size", r.tp_size},
           {"pp_size", r.pp_size}};
}

inline void from_json(const json &j, PullDiffusionModelRequest &r) {
  // j.at("file_names").get_to(r.file_names);
  j.at("file_name").get_to(r.file_name);
}

inline void to_json(json &j, const PullDiffusionModelRequest &r) {
  // j = json{{"file_names", r.file_names}};
  j = json{{"file_name", r.file_name}};
}

inline void from_json(const json &j, PullModelResponse &r) {
  j.at("task_id").get_to(r.task_id);
}

inline void to_json(json &j, const PullModelResponse &r) {
  j = json{{"task_id", r.task_id}};
}

inline void from_json(const json &j, CheckModelRequest &r) {
  j.at("model_name").get_to(r.model_name);
  j.at("task_id").get_to(r.task_id);
}

inline void to_json(json &j, const CheckModelRequest &r) {
  j = json{{"model_name", r.model_name}, {"task_id", r.task_id}};
}

inline void from_json(const json &j, CheckModelResponse &r) {
  j.at("done").get_to(r.done);
}

inline void to_json(json &j, const CheckModelResponse &r) {
  j = json{{"done", r.done}};
}

inline void from_json(const json &j, LoadTensorRequest &r) {
  j.at("tensor_name").get_to(r.tensor_name);
  j.at("tensor_size").get_to(r.tensor_size);
  j.at("rank").get_to(r.rank);
}

inline void to_json(json &j, const LoadTensorRequest &r) {
  j = json{{"tensor_name", r.tensor_name},
           {"tensor_size", r.tensor_size},
           {"rank", r.rank}};
}

inline void from_json(const json &j, LoadTensorResponse &r) {
  auto arr = j.at("handler").get<vector<uint8_t>>();
  if (arr.size() != sizeof(cudaIpcMemHandle_t))
    throw runtime_error("Invalid handler size");
  memcpy(&r.handler, arr.data(), sizeof(cudaIpcMemHandle_t));

  j.at("offset").get_to(r.offset);
  j.at("loaded_size").get_to(r.loaded_size);
  j.at("resize_tensor").get_to(r.resize_tensor);
}

inline void to_json(json &j, const LoadTensorResponse &r) {
  vector<uint8_t> arr(sizeof(cudaIpcMemHandle_t));
  memcpy(arr.data(), &r.handler, sizeof(cudaIpcMemHandle_t));
  j = json{{"handler", arr},
           {"offset", r.offset},
           {"loaded_size", r.loaded_size},
           {"resize_tensor", r.resize_tensor}};
}

inline void from_json(const json &j, RevertHandlerRequest &r) {
  j.at("tensor_name").get_to(r.tensor_name);
  j.at("tensor_size").get_to(r.tensor_size);
  j.at("rank").get_to(r.rank);
}

inline void to_json(json &j, const RevertHandlerRequest &r) {
  j = json{{"tensor_name", r.tensor_name},
           {"tensor_size", r.tensor_size},
           {"rank", r.rank}};
}

inline void from_json(const json &j, RevertHandlerResponse &r) {
  j.at("success").get_to(r.success);
}

inline void to_json(json &j, const RevertHandlerResponse &r) {
  j = json{{"success", r.success}};
}

inline void from_json(const json &j, ResetStatusRequest &r) {
  j.at("rank").get_to(r.rank);
}

inline void to_json(json &j, const ResetStatusRequest &r) {
  j = json{{"rank", r.rank}};
}
inline void from_json(const json &j, EmptyRequestResponse &r) {}

inline void to_json(json &j, const EmptyRequestResponse &r) { j = json{}; }
