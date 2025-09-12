cmake_minimum_required(VERSION 3.15...4.0)
project(CppMemoryManager LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# add_compile_options(-w)

option(BUILD_TESTS "whether to build tests" OFF)

list(APPEND CMAKE_PREFIX_PATH "/usr/local/lib/python3.10/dist-packages/torch/share/cmake")
list(APPEND CMAKE_PREFIX_PATH "${TORCH_INSTALL_PREFIX}/share/cmake/Torch")
list(APPEND CMAKE_PREFIX_PATH "/root/.local")
find_package(Torch REQUIRED)
find_package(fmt REQUIRED)
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
if(BUILD_PYBIND)
    find_package(pybind11 CONFIG REQUIRED)
    set(PYBIND11_PYTHON_VERSION 3.10)
endif()

# pybinding config variables
set(CMAKE_CXX_STANDARD 17)
set(TORCH_DISABLE_PROFILER ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_BUILD_TYPE Debug)
add_subdirectory(third_party/spdlog)
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated)

# ----Engine----
set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GEN_DIR})

set(PROTO_DIR ${CMAKE_SOURCE_DIR}/proto)
set(PROTO_SRC ${CMAKE_SOURCE_DIR}/proto/generate.proto)

find_program(GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
if(NOT GRPC_CPP_PLUGIN_EXECUTABLE)
    message(FATAL_ERROR "Cannot find grpc_cpp_plugin")
endif()

add_custom_command(
    OUTPUT
    ${GEN_DIR}/generate.pb.cc
    ${GEN_DIR}/generate.pb.h
    ${GEN_DIR}/generate.grpc.pb.cc
    ${GEN_DIR}/generate.grpc.pb.h
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GEN_DIR}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    --proto_path=${PROTO_DIR}
    --cpp_out=${GEN_DIR}
    --grpc_out=${GEN_DIR}
    --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}
    ${PROTO_SRC}
    DEPENDS ${PROTO_SRC}
    VERBATIM
)

add_library(textgen_proto "${GEN_DIR}/generate.pb.cc" "${GEN_DIR}/generate.grpc.pb.cc")
target_include_directories(textgen_proto PUBLIC "${GEN_DIR}")
target_link_libraries(textgen_proto PUBLIC gRPC::grpc++ protobuf::libprotobuf)
target_compile_options(textgen_proto PRIVATE -fPIC)

add_executable(server src/server.cc src/blitz_engine.cc)
target_include_directories(server PUBLIC "${GEN_DIR}" ${TORCH_INCLUDE_DIRS})
target_link_libraries(server PUBLIC textgen_proto ${Protobuf_LIBRARIES} ${TORCH_LIBRARIES} spdlog::spdlog_header_only)

# ----C++ Test----
# if(BUILD_TESTS)
#     # add_executable(test_engine tests/test_engine.cc src/blitz_engine.cc)
#     # target_link_libraries(test_engine PUBLIC ${TORCH_LIBRARIES} spdlog::spdlog_header_only)
#     add_executable(test_engine tests/test_engine_alloc.cc src/blitz_engine.cc)
#     target_link_libraries(test_engine PUBLIC ${TORCH_LIBRARIES} spdlog::spdlog_header_only)
#     add_executable(child_process tests/test_engine_child.cc src/blitz_engine.cc)
#     target_link_libraries(child_process PUBLIC ${TORCH_LIBRARIES} spdlog::spdlog_header_only)
#     add_executable(client tests/grpc_client.cc)
#     target_include_directories(client PUBLIC "${GEN_DIR}" ${TORCH_INCLUDE_DIRS})
#     target_link_libraries(client PUBLIC textgen_proto ${Protobuf_LIBRARIES} ${TORCH_LIBRARIES} spdlog::spdlog_header_only)
#     # target_include_directories(test_engine PRIVATE ${TORCH_INCLUDE_DIRS})
#     # target_link_libraries(test_engine PRIVATE "${TORCH_LIBRARIES}" spdlog::spdlog_header_only)

#     # add_library(blitz_header INTERFACE)
#     # target_include_directories(blitz_header INTERFACE ${TORCH_INCLUDE_DIRS})
# endif()

# ----PYBinding----
# if(BUILD_PYBIND)
#     add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
#     pybind11_add_module(blitz_lib python/blitz_lib/py_binding.cpp)
#     target_link_libraries(blitz_lib PRIVATE blitz_header ${TORCH_LIBRARIES} spdlog::spdlog_header_only pybind11::module)

#     find_library(TORCH_PYTHON_LIBRARY
#         NAMES torch_python
#         HINTS
#         "${TORCH_INSTALL_PREFIX}/lib"
#         "${TORCH_INSTALL_PREFIX}"
#         "${TORCH_LIBRARY_DIRS}"
#         "${CMAKE_CURRENT_SOURCE_DIR}/venv/Lib/site-packages/torch/lib"
#     )

#     if(TORCH_PYTHON_LIBRARY)
#         message(STATUS "Found torch_python: ${TORCH_PYTHON_LIBRARY}")
#         target_link_libraries(blitz_lib
#             PRIVATE
#             ${TORCH_PYTHON_LIBRARY}
#         )
#     else()
#         message(
#             WARNING
#             "Could not find torch_python.lib; you may still have link errors."
#         )
#     endif()


#     install(TARGETS blitz_lib
#         LIBRARY DESTINATION python/blitz_lib)

# endif()
