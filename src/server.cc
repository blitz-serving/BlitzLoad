#include <blitz_engine.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <openssl/sha.h>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "generate.grpc.pb.h"
#include "generate.pb.h"
#include "spdlog/spdlog.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

namespace fs = std::filesystem;

void UnsetProxyEnv() {
  const char *proxy_vars[] = {"HTTP_PROXY",  "http_proxy", "HTTPS_PROXY",
                              "https_proxy", "ALL_PROXY",  "all_proxy",
                              "NO_PROXY",    "no_proxy"};
  for (const char *var : proxy_vars) {
    if (std::getenv(var)) {
      if (unsetenv(var) != 0) {
        std::cerr << "Failed to unset " << var << std::endl;
      }
    }
  }
  std::cout << "All proxy environment variables have been unset." << std::endl;
}

namespace generate {
namespace v2 {

class ParamServiceImpl final : public ParamService::Service {
public:
  ParamServiceImpl(std::vector<int> &visible_devices) {
    engine_ptr = std::make_unique<blitz::BlitzEngine>(
        visible_devices, (size_t)512 * 1024 * 1024);
  }

  Status PullModel(ServerContext *ctx, const PullModelRequest *req,
                   PullModelResponse *resp) override {
    (void)ctx;
    (void)resp;

    spdlog::info("Want to load model: {}", req->model_name());
    auto bin_files = std::vector<std::string>();
    std::regex pattern(R"(dangertensors\.\d+\.bin)");
    try {
      for (const auto &entry : fs::directory_iterator(req->model_name())) {
        if (entry.is_regular_file()) {
          std::string filename = entry.path().filename().string();
          if (std::regex_match(filename, pattern)) {
            bin_files.push_back(entry.path().string());
            spdlog::info("Found file: {}", entry.path().string());
          }
        }
      }
    } catch (const std::exception &e) {
      spdlog::error("Error: {}", e.what());
    }
    std::string model_name = req->model_name();
    std::string data = req->model_name() + std::to_string(time(nullptr));
    auto id = sha256(data);
    task_map[id] = false;
    std::thread([this, bin_files, id, model_name] {
      auto rank_num = engine_ptr->pull_model(model_name);
      engine_ptr->mem_to_buffer(model_name, rank_num);
      task_map[id] = true;
    }).detach();
    resp->set_task_id(id);
    spdlog::info("Task ID is {}", id);

    return Status::OK;
  }

  Status CheckModel(ServerContext *ctx, const CheckModelRequest *req,
                    CheckModelResponse *resp) override {
    auto res = task_map[req->task_id()];
    resp->set_done(res);
    // spdlog::info("Model loaded? {}, task_id {}", res, req->task_id());

    return Status::OK;
  }

  /// deprecated
  Status LoadWeight(ServerContext *ctx, const LoadWeightRequest *req,
                    LoadWeightResponse *resp) override {
    (void)ctx;

    cudaIpcMemHandle_t handle;
    const std::string &serialized_handle = req->ipc_handle();
    memcpy(&handle, serialized_handle.data(), sizeof(handle));
    engine_ptr->mem_to_tensor(handle, req->tensor_name(), req->tensor_size(),
                              req->tensor_device());

    // engine_ptr->buffer_to_tensor(handle, req->tensor_device(),
    //                              req->tensor_size());

    return Status::OK;
  }

  Status GetHandler(ServerContext *ctx, const GetHandlerRequest *req,
                    GetHandlerResponse *resp) override {
    (void)ctx;
    auto start = std::chrono::steady_clock::now();
    cudaIpcMemHandle_t handle;
    size_t offset = 0;
    // FIXME: hard code shard_id = 0
    auto rank_ = req->rank();
    auto loaded_size =
        engine_ptr->export_handler(&handle, &offset, req->tensor_size(), rank_);
    resp->set_ipc_handler(reinterpret_cast<const char *>(&handle),
                          sizeof(handle));
    resp->set_offset(offset);
    resp->set_loaded_size(loaded_size);
    auto end = std::chrono::steady_clock::now();
    auto elapse_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    elaspse_all += elapse_ms;

    spdlog::info("cum time: {}ms", elaspse_all);
    return Status::OK;
  }

  // FIXME: hard code shard_id = 0
  Status RevertHandler(ServerContext *ctx, const RevertHandlerRequest *req,
                       RevertHandlerResponse *resp) override {
    auto start = std::chrono::steady_clock::now();
    auto rank_ = req->rank();
    engine_ptr->free_handler(req->tensor_size(), rank_);
    auto end = std::chrono::steady_clock::now();
    auto elapse_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    elaspse_all2 += elapse_ms;

    spdlog::info("cum time: {}ms", elaspse_all2);
    return Status::OK;
  }

  std::string sha256(const std::string &str) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256((unsigned char *)str.c_str(), str.size(), hash);

    std::ostringstream oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i)
      oss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    return oss.str();
  }

  void set_health_check_service(
      grpc::HealthCheckServiceInterface *health_check_service) {
    health_check_service_ = health_check_service;
  }

private:
  grpc::HealthCheckServiceInterface *health_check_service_ = nullptr;
  std::unique_ptr<blitz::BlitzEngine> engine_ptr;
  std::map<std::string, bool> task_map;
  size_t elaspse_all = 0;
  size_t elaspse_all2 = 0;
};

} // namespace v2
} // namespace generate

static void RunServer(const std::string &addr) {
  std::vector<int> devices = {0};

  generate::v2::ParamServiceImpl service(devices);
  grpc::EnableDefaultHealthCheckService(true);
  ServerBuilder builder;
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  std::unique_ptr<Server> server(builder.BuildAndStart());
  service.set_health_check_service(server->GetHealthCheckService());
  std::cout << "TextGenerationService listening on " << addr << std::endl;
  server->Wait();
}

int main(int argc, char **argv) {
  UnsetProxyEnv();
  std::string addr = "unix:///tmp/grpc.sock";
  if (argc >= 2) {
    addr = argv[1];
  }
  RunServer(addr);
  return 0;
}
