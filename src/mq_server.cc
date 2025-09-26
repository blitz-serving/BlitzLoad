#include "logger.h"
#include "spdlog/spdlog.h"
#include <blitz_engine.h>
#include <chrono>
#include <common_tools.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mqmsg_structs.hpp>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <zmq.h>
#include <zmq.hpp>

using nlohmann::json;

class Mq_Server {
public:
  Mq_Server(std::vector<int> devices, size_t buffer_size) {
    engine_ptr = std::make_unique<blitz::BlitzEngine>(devices, buffer_size);
    task_map = std::make_unique<std::map<string, bool>>();
  }

  void run() {
    zmq::context_t ctx(1);

    zmq::socket_t pull_model_socket(ctx, zmq::socket_type::rep);
    pull_model_socket.bind("tcp://*:55555");

    zmq::socket_t check_model_socket(ctx, zmq::socket_type::rep);
    check_model_socket.bind("tcp://*:55556");

    zmq::socket_t load_socket(ctx, zmq::socket_type::rep);
    load_socket.bind("tcp://*:55557");

    zmq::socket_t revert_socket(ctx, zmq::socket_type::rep);
    revert_socket.bind("tcp://*:55558");

    zmq::pollitem_t items[] = {
        {pull_model_socket, 0, ZMQ_POLLIN, 0},
        {check_model_socket, 0, ZMQ_POLLIN, 0},
        {load_socket, 0, ZMQ_POLLIN, 0},
        {revert_socket, 0, ZMQ_POLLIN, 0},
    };

    while (true) {
      zmq::poll(items, 4, std::chrono::milliseconds(-1));

      if (items[0].revents & ZMQ_POLLIN) {
        zmq::message_t msg;
        auto result = pull_model_socket.recv(msg);

        if (result) {
          PullModelRequest req = json::parse(to_string(msg));
          auto id = gen_sha256(req.model_name);
          (*task_map)[id] = false;
          std::thread([req, id, this] {
            auto rank_num = engine_ptr->pull_model(req.model_name);
            engine_ptr->mem_to_buffer(req.model_name, rank_num);
            LOG_ASSERT(rank_num == req.world_size,
                       "Dangertensor num {} != world size {}", rank_num,
                       req.world_size);
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
          auto loaded_size = engine_ptr->export_handler(
              &handle, &offset, req.tensor_size, req.rank);

          LoadTensorResponse resp{handle, offset, loaded_size};
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
    }
  }

private:
  std::unique_ptr<blitz::BlitzEngine> engine_ptr;
  std::unique_ptr<std::map<std::string, bool>> task_map;
};

int main() {
  auto server = Mq_Server({0, 1, 2}, 512 * 1024 * 1024);
  server.run();
  return 0;
}