cmake_minimum_required(VERSION 3.10)
project(CppMemoryManager LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Debug)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
list(APPEND CMAKE_PREFIX_PATH "/usr/local/lib/python3.10/dist-packages/torch/share/cmake")
list(APPEND CMAKE_PREFIX_PATH "${TORCH_INSTALL_PREFIX}/share/cmake/Torch")

find_package(Torch REQUIRED)
find_package(fmt REQUIRED)
add_subdirectory(third_party/spdlog)
include_directories(${CMAKE_SOURCE_DIR}/include)


add_executable(test_data_mover tests/test_data_mover.cc)
target_link_libraries(test_data_mover PRIVATE "${TORCH_LIBRARIES}" spdlog::spdlog)
target_include_directories(test_data_mover PRIVATE ${TORCH_INCLUDE_DIRS})
