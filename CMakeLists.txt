cmake_minimum_required(VERSION 3.15...4.0)
project(CppMemoryManager LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# add_compile_options(-w)

list(APPEND CMAKE_PREFIX_PATH "/usr/local/lib/python3.10/dist-packages/torch/share/cmake")
list(APPEND CMAKE_PREFIX_PATH "${TORCH_INSTALL_PREFIX}/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(fmt REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# pybinding config variables
set(CMAKE_CXX_STANDARD 17)
set(TORCH_DISABLE_PROFILER ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PYBIND11_PYTHON_VERSION 3.10)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

set(CMAKE_BUILD_TYPE Debug)
add_subdirectory(third_party/spdlog)
include_directories(${CMAKE_SOURCE_DIR}/include)

# ----C++ Test----
add_executable(test_data_mover tests/test_data_mover.cc)
target_link_libraries(test_data_mover PRIVATE "${TORCH_LIBRARIES}" spdlog::spdlog_header_only)
target_include_directories(test_data_mover PRIVATE ${TORCH_INCLUDE_DIRS})

add_library(blitz_header INTERFACE)
target_include_directories(blitz_header INTERFACE ${TORCH_INCLUDE_DIRS})


# ----PYBinding----
pybind11_add_module(blitz_lib python/blitz_lib/py_binding.cpp)
target_link_libraries(blitz_lib PRIVATE blitz_header ${TORCH_LIBRARIES} spdlog::spdlog_header_only pybind11::module)

find_library(TORCH_PYTHON_LIBRARY
    NAMES torch_python
    HINTS
      "${TORCH_INSTALL_PREFIX}/lib"
      "${TORCH_INSTALL_PREFIX}"
      "${TORCH_LIBRARY_DIRS}"
      "${CMAKE_CURRENT_SOURCE_DIR}/venv/Lib/site-packages/torch/lib"
)

if (TORCH_PYTHON_LIBRARY)
    message(STATUS "Found torch_python: ${TORCH_PYTHON_LIBRARY}")
    target_link_libraries(blitz_lib
      PRIVATE
        ${TORCH_PYTHON_LIBRARY}
    )
    else()
    message(
        WARNING 
        "Could not find torch_python.lib; you may still have link errors."
    )
endif()


install(TARGETS blitz_lib
        LIBRARY DESTINATION python/blitz_lib)