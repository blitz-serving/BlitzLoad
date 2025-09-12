#pragma once

#include <cassert>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#define LOG_ASSERT(cond, ...)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      spdlog::error(__VA_ARGS__);                                              \
      assert(cond);                                                            \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      throw std::runtime_error(std::string("CUDA: ") +                         \
                               cudaGetErrorString(_err));                      \
    }                                                                          \
  } while (0)