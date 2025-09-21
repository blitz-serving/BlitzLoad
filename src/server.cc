#include <blitz_engine.h>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <grpcpp/grpcpp.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <iostream>
#include <memory>
#include <regex>
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
    auto shard_num = engine_ptr->ssd_to_mem(bin_files);
    engine_ptr->mem_to_buffer(shard_num);

    return Status::OK;
  }

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
    cudaIpcMemHandle_t handle;
    size_t offset = 0;
    // FIXME: hard code shard_id = 0
    auto loaded_size =
        engine_ptr->export_handler(&handle, &offset, req->tensor_size(), 0);
    resp->set_ipc_handler(reinterpret_cast<const char *>(&handle),
                          sizeof(handle));
    resp->set_offset(offset);
    resp->set_loaded_size(loaded_size);
    return Status::OK;
  }

  // FIXME: hard code shard_id = 0
  Status RevertHandler(ServerContext *ctx, const RevertHandlerRequest *req,
                       RevertHandlerResponse *resp) override {
    engine_ptr->free_handler(req->tensor_size(), 0);
    return Status::OK;
  }

private:
  std::unique_ptr<blitz::BlitzEngine> engine_ptr;
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
  std::cout << "TextGenerationService listening on " << addr << std::endl;
  server->Wait();
}

int main(int argc, char **argv) {
  UnsetProxyEnv();
  std::string addr = "0.0.0.0:60060";
  if (argc >= 2) {
    addr = argv[1];
  }
  RunServer(addr);
  return 0;
}
