#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <torch/extension.h>

void my_copy_to_gpu(torch::Tensor tensor, uintptr_t src_ptr, size_t size);

uintptr_t alloc_pinned_memory(size_t size);

void free_pinned_memory(uintptr_t ptr);