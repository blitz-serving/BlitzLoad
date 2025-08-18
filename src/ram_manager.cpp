// #include <cuda_runtime.h>
// #include <iostream>
// #include <pybind11/pybind11.h>
// #include <stdexcept>
// #include <torch/extension.h>

#include "ram_manager.h"

// 从 CPU 数据拷贝到 GPU tensor
void my_copy_to_gpu(torch::Tensor tensor, uintptr_t src_ptr, size_t size) {
  TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA");
  TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
  TORCH_CHECK(tensor.nbytes() >= size, "Size mismatch");

  void *dst = tensor.data_ptr();
  void *src = reinterpret_cast<void *>(src_ptr);

  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
  TORCH_CHECK(err == cudaSuccess,
              "cudaMemcpy failed: ", cudaGetErrorString(err));
}

// 分配 pinned memory
uintptr_t alloc_pinned_memory(size_t size) {
  void *ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
  TORCH_CHECK(err == cudaSuccess,
              "cudaHostAlloc failed: ", cudaGetErrorString(err));
  return reinterpret_cast<uintptr_t>(ptr);
}

// 释放 pinned memory
void free_pinned_memory(uintptr_t ptr) {
  cudaError_t err = cudaFreeHost(reinterpret_cast<void *>(ptr));
  TORCH_CHECK(err == cudaSuccess,
              "cudaFreeHost failed: ", cudaGetErrorString(err));
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("copy_to_gpu", &my_copy_to_gpu, "Copy CPU data to GPU tensor");
//   m.def("alloc_pinned_memory", &alloc_pinned_memory,
//         "Allocate pinned (page-locked) host memory");
//   m.def("free_pinned_memory", &free_pinned_memory,
//         "Free pinned (page-locked) host memory");
// }
