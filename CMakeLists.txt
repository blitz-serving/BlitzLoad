cmake_minimum_required(VERSION 3.15...4.0)
project(CppMemoryManager LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# add_definitions(-DSPDLOG_FMT_EXTERNAL=0)
# add_compile_options(-w)

option(BUILD_TESTS "whether to build tests" OFF)

list(APPEND CMAKE_PREFIX_PATH "/usr/local/lib/python3.10/dist-packages/torch/share/cmake")
list(APPEND CMAKE_PREFIX_PATH "${TORCH_INSTALL_PREFIX}/share/cmake/Torch")
list(APPEND CMAKE_PREFIX_PATH "/root/.local")
find_package(Torch REQUIRED)
find_package(fmt REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
# find_package(Protobuf REQUIRED)
# find_package(gRPC REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PC_ZMQ REQUIRED libzmq)

# pybinding config variables
set(CMAKE_CXX_STANDARD 17)
set(TORCH_DISABLE_PROFILER ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_BUILD_TYPE Debug)
add_subdirectory(third_party/spdlog)
# add_subdirectory(third_party/cppzmq)

include_directories(BEFORE ${CMAKE_SOURCE_DIR}/third_party/spdlog/include ${CMAKE_SOURCE_DIR}/third_party/cppzmq ${CMAKE_SOURCE_DIR}/third_party/json/single_include)
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated)
# add_library(nlohmann_json INTERFACE)
# target_include_directories(nlohmann_json INTERFACE
#     ${CMAKE_SOURCE_DIR}/third_party/json/single_include
# )


# ----Engine----
set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GEN_DIR})

set(PROTO_DIR ${CMAKE_SOURCE_DIR}/proto)
set(PROTO_SRC ${CMAKE_SOURCE_DIR}/proto/generate.proto)

find_program(GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
if(NOT GRPC_CPP_PLUGIN_EXECUTABLE)
    message(FATAL_ERROR "Cannot find grpc_cpp_plugin")
endif()

# add_custom_command(
#     OUTPUT
#     ${GEN_DIR}/generate.pb.cc
#     ${GEN_DIR}/generate.pb.h
#     ${GEN_DIR}/generate.grpc.pb.cc
#     ${GEN_DIR}/generate.grpc.pb.h
#     COMMAND ${CMAKE_COMMAND} -E make_directory ${GEN_DIR}
#     COMMAND ${Protobuf_PROTOC_EXECUTABLE}
#     --proto_path=${PROTO_DIR}
#     --cpp_out=${GEN_DIR}
#     --grpc_out=${GEN_DIR}
#     --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}
#     ${PROTO_SRC}
#     DEPENDS ${PROTO_SRC}
#     VERBATIM
# )

# add_library(textgen_proto "${GEN_DIR}/generate.pb.cc" "${GEN_DIR}/generate.grpc.pb.cc")
# target_include_directories(textgen_proto PUBLIC "${GEN_DIR}")
# target_link_libraries(textgen_proto PUBLIC gRPC::grpc++ protobuf::libprotobuf)
# target_compile_options(textgen_proto PRIVATE -fPIC)

# add_executable(server src/server.cc src/blitz_engine.cc)
# target_include_directories(server PUBLIC "${GEN_DIR}" ${TORCH_INCLUDE_DIRS})
# target_link_libraries(server PUBLIC textgen_proto ${Protobuf_LIBRARIES} ${TORCH_LIBRARIES})



add_custom_command(
    OUTPUT mq_types.py
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utils/gen_type.py
            ${CMAKE_SOURCE_DIR}/include/mqmsg_structs.hpp
            > ${CMAKE_SOURCE_DIR}/tests/mq_types.py
    DEPENDS ${CMAKE_SOURCE_DIR}/include/mqmsg_structs.hpp
    VERBATIM
)
add_custom_target(generate_py ALL DEPENDS mq_types.py)

add_executable(mq_server src/mq_server.cc src/blitz_engine.cc)
target_include_directories(mq_server PUBLIC ${TORCH_INCLUDE_DIRS} ${PC_ZMQ_INCLUDE_DIRS})
target_link_libraries(mq_server PUBLIC crypto ${TORCH_LIBRARIES} ${PC_ZMQ_LIBRARIES})