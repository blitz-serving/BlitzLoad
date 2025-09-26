
#pragma once
#include "zmq.hpp"
#include <cstring>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <string>

using namespace std;
using nlohmann::json;

inline std::string to_string(zmq::message_t &msg) {
  return string(static_cast<char *>(msg.data()), msg.size());
}

inline std::string gen_sha256(const std::string model_name) {
  std::string str = model_name + std::to_string(time(nullptr));
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256((unsigned char *)str.c_str(), str.size(), hash);

  std::ostringstream oss;
  for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i)
    oss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
  return oss.str();
}

inline zmq::message_t build_msg(json object) {
  auto payload = object.dump();
  zmq::message_t msg(payload.size());
  memcpy(msg.data(), payload.data(), payload.size());
  return msg;
}