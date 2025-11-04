#include "logger.h"
#include "spdlog/fmt/bundled/format.h"
#include "spdlog/spdlog.h"
#include <atomic>
#include <blitz_engine.h>
#include <chrono>
#include <common_tools.hpp>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <mqmsg_structs.hpp>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <zmq.h>
#include <zmq.hpp>

using nlohmann::json;

std::atomic<bool> running{true};

void signal_handler(int) { running = false; }

class Mq_Server {
public:
  Mq_Server(std::vector<int> devices, size_t buffer_size, int port = 55555) {
    engine_ptr = std::make_unique<blitz::BlitzEngine>(devices, buffer_size);
    task_map = std::make_unique<std::map<string, bool>>();
    this->port = port;
  }

  void run() {
    zmq::context_t ctx(1);

    zmq::socket_t pull_model_socket(ctx, zmq::socket_type::rep);
    pull_model_socket.bind(fmt::format("tcp://*:{}", port));

    zmq::socket_t check_model_socket(ctx, zmq::socket_type::rep);
    check_model_socket.bind(fmt::format("tcp://*:{}", port + 1));

    zmq::socket_t load_socket(ctx, zmq::socket_type::rep);
    load_socket.bind(fmt::format("tcp://*:{}", port + 2));

    zmq::socket_t revert_socket(ctx, zmq::socket_type::rep);
    revert_socket.bind(fmt::format("tcp://*:{}", port + 3));

    zmq::socket_t reset_socket(ctx, zmq::socket_type::rep);
    reset_socket.bind(fmt::format("tcp://*:{}", port + 4));

    zmq::socket_t pull_model_diffusion_socket(ctx, zmq::socket_type::rep);
    pull_model_diffusion_socket.bind(fmt::format("tcp://*:{}", port + 5));

    zmq::socket_t meta_socket(ctx, zmq::socket_type::rep);
    meta_socket.bind(fmt::format("tcp://*:{}", port + 6));

    zmq::socket_t meta_tensor_socket(ctx, zmq::socket_type::rep);
    meta_tensor_socket.bind(fmt::format("tcp://*:{}", port + 7));

    zmq::pollitem_t items[] = {{pull_model_socket, 0, ZMQ_POLLIN, 0},
                               {check_model_socket, 0, ZMQ_POLLIN, 0},
                               {load_socket, 0, ZMQ_POLLIN, 0},
                               {revert_socket, 0, ZMQ_POLLIN, 0},
                               {reset_socket, 0, ZMQ_POLLIN, 0},
                               {pull_model_diffusion_socket, 0, ZMQ_POLLIN, 0},
                               {meta_socket, 0, ZMQ_POLLIN, 0},
                               {meta_tensor_socket, 0, ZMQ_POLLIN, 0}};

    while (running) {
      zmq::poll(items, 8, std::chrono::milliseconds(-1));

      if (items[0].revents & ZMQ_POLLIN) {
        zmq::message_t msg;
        auto result = pull_model_socket.recv(msg);

        if (result) {
          spdlog::info("Pull model request");
          PullModelRequest req = json::parse(to_string(msg));
          auto id = gen_sha256(req.model_name);
          (*task_map)[id] = false;
          std::thread([req, id, this] {
            int tp_size = req.tp_size, pp_size = req.pp_size;
            auto rank_num =
                engine_ptr->pull_model(req.model_name, tp_size, pp_size);
            LOG_ASSERT(rank_num == req.world_size,
                       "Dangertensor num {} != world size {}", rank_num,
                       req.world_size);
            auto danger_tensor_index_name =
                gen_dangertensor_index_name(req.model_name, tp_size, pp_size);
            task2_model_info[id] = {danger_tensor_index_name, rank_num};

            (*task_map)[id] = true;
          }).detach();

          PullModelResponse resp{id};
          auto reply = build_msg(resp);
          pull_model_socket.send(reply, zmq::send_flags::none);

        } else {
          spdlog::error("Receive Pull Model Req Failed");
        }
      }

      if (items[1].revents & ZMQ_POLLIN) {
        zmq::message_t msg;
        auto result = check_model_socket.recv(msg);
        if (result) {
          CheckModelRequest req = json::parse(to_string(msg));
          CheckModelResponse resp{(*task_map)[req.task_id]};
          try {
            if ((*task_map)[req.task_id] &&
                task2_model_info.count(req.task_id)) {
              auto [danger_tensor_index_name, rank_num] =
                  task2_model_info[req.task_id];
              spdlog::info("Trigger mem_to_buffer of {}",
                           danger_tensor_index_name);
              engine_ptr->mem_to_buffer(danger_tensor_index_name, rank_num);
              // to avoid re-trigger mem_to_buffer
              task2_model_info.erase(req.task_id);
            }
          } catch (const std::exception &e) {
            spdlog::error("Error in mem_to_buffer: {}", e.what());
          }
          auto reply = build_msg(resp);
          check_model_socket.send(reply, zmq::send_flags::none);
        } else {
          spdlog::error("Receive Check Model Req Failed");
        }
      }

      if (items[2].revents & ZMQ_POLLIN) {
        zmq::message_t msg;
        auto result = load_socket.recv(msg);
        if (result) {
          LoadTensorRequest req = json::parse(to_string(msg));
          cudaIpcMemHandle_t handle;
          size_t offset;
          auto [loaded_size, resize_tensor] = engine_ptr->export_handler(
              &handle, &offset, req.tensor_size, req.rank);

          LoadTensorResponse resp{handle, offset, loaded_size, resize_tensor};
          auto reply = build_msg(resp);
          load_socket.send(reply, zmq::send_flags::none);
        } else {
          spdlog::error("Receive Load Tensor Req Failed");
        }
      }

      if (items[3].revents & ZMQ_POLLIN) {
        zmq::message_t msg;
        auto result = revert_socket.recv(msg);
        if (result) {
          RevertHandlerRequest req = json::parse(to_string(msg));
          engine_ptr->free_handler(req.tensor_size, req.rank);

          RevertHandlerResponse resp{true};
          auto reply = build_msg(resp);
          revert_socket.send(reply, zmq::send_flags::none);
        } else {
          spdlog::error("Receive Revert Tensor Req Failed");
        }
      }

      if (items[4].revents & ZMQ_POLLIN) {
        zmq::message_t msg;
        auto result = reset_socket.recv(msg);
        if (result) {
          ResetStatusRequest req = json::parse(to_string(msg));
          engine_ptr->reset_status(req.rank);

          EmptyRequestResponse resp{};
          auto reply = build_msg(resp);
          reset_socket.send(reply, zmq::send_flags::none);
        } else {
          spdlog::error("Receive Revert Tensor Req Failed");
        }
      }

      if (items[5].revents & ZMQ_POLLIN) {
        zmq::message_t msg;
        auto result = pull_model_diffusion_socket.recv(msg);

        if (result) {
          spdlog::info("Pull model diffusion request");
          PullDiffusionModelRequest req = json::parse(to_string(msg));
          auto id = gen_sha256(req.file_name);
          (*task_map)[id] = false;
          std::thread([req, id, this] {
            auto dangertensor_index_name =
                gen_dangertensor_index_name(req.file_name, 1, 1);
            task2_model_info[id] = {dangertensor_index_name, 1};
            engine_ptr->load_file_to_mem(req.file_name, 0,
                                         dangertensor_index_name);
            (*task_map)[id] = true;
          }).detach();

          PullModelResponse resp{id};
          auto reply = build_msg(resp);
          pull_model_diffusion_socket.send(reply, zmq::send_flags::none);
        } else {
          spdlog::error("Receive Pull Model Diffusion Req Failed");
        }
      }
      if (items[6].revents & ZMQ_POLLIN) {
        zmq::message_t msg;
        auto result = meta_socket.recv(msg);

        if (result) {
          spdlog::info("Meta Info request");
          GetMetaRequest req = json::parse(to_string(msg));
          auto dangertensor_index_name =
              gen_dangertensor_index_name(req.file_name, 1, 1);
          std::string meta_str = "";
          engine_ptr->export_meta(dangertensor_index_name, 1, meta_str);
          GetMetaResponse resp{meta_str};
          auto reply = build_msg(resp);
          meta_socket.send(reply, zmq::send_flags::none);
        } else {
          spdlog::error("Receive Meta Req Failed");
        }
      }

      if (items[7].revents & ZMQ_POLLIN) {
        zmq::message_t msg;
        auto result = meta_tensor_socket.recv(msg);

        if (result) {
          spdlog::info("Meta tensor request");
          GetMetaTensorRequest req = json::parse(to_string(msg));
          auto dangertensor_index_name =
              gen_dangertensor_index_name(req.file_name, 1, 1);
          auto meta_tensors =
              engine_ptr->export_meta_tensors(dangertensor_index_name, 1);
          GetMetaTensorResponse resp{meta_tensors};
          auto reply = build_msg(resp);
          meta_tensor_socket.send(reply, zmq::send_flags::none);
        } else {
          spdlog::error("Receive Meta Tensor Req Failed");
        }
      }
    }
  }

private:
  std::unique_ptr<blitz::BlitzEngine> engine_ptr;
  std::unique_ptr<std::map<std::string, bool>> task_map;
  std::map<std::string, std::pair<std::string, int>> task2_model_info;
  int port;
  // std::atomic<int> load_revert_cnt = 0;
};

std::vector<int> parse_devices(const std::string &devices_str) {
  std::vector<int> devices;
  std::stringstream ss(devices_str);
  std::string token;
  while (std::getline(ss, token, ',')) {
    try {
      devices.push_back(std::stoi(token));
    } catch (const std::invalid_argument &) {
      std::cerr << "Invalid device id: " << token << std::endl;
      exit(1);
    }
  }
  return devices;
}

int main(int argc, char *argv[]) {
  std::vector<int> devices;
  int port = 55555;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--devices") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --devices requires a value (e.g., --devices 0,1,2)"
                  << std::endl;
        return 1;
      }
      devices = parse_devices(argv[++i]);
    } else if (arg.rfind("--devices=", 0) == 0) {
      devices = parse_devices(arg.substr(10));
    }

    if (arg == "--port") {
      // start port, default port is 55555
      try {
        port = stoi(argv[++i]);
      } catch (const std::invalid_argument &) {
        std::cerr << "Invalid port number: " << argv[i] << std::endl;
        return 1;
      }
    }
  }

  if (devices.empty()) {
    std::cout << "No devices specified. Use --devices 0,1,2 to select GPUs.\n";
    return 0;
  }

  auto server = Mq_Server(devices, 512 * 1024 * 1024, port);
  server.run();
  return 0;
}