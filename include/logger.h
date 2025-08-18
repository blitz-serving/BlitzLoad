#pragma once

#include <cassert>
#include <spdlog/spdlog.h>

#define LOG_ASSERT(cond, ...)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      spdlog::error(__VA_ARGS__);                                              \
      assert(cond);                                                            \
    }                                                                          \
  } while (0)