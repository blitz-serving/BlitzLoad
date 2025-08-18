#include "data_mover.hpp"
#include <c10/core/DeviceType.h>
#include <c10/util/BFloat16.h>

int main() {
  auto data_mover = DataMover();
  data_mover.init(Mode::CPU);
  auto buffer = data_mover.register_buffer(16ULL * 1024 * 1024 * 1024, "blitz");
  data_mover.load_file_to_buffer_sync("blitz",
                                      {"/nvme/models/Meta-Llama-3-8B-Instruct/"
                                       "model-00001-of-00004.safetensors"});
  torch::Tensor t = torch::zeros(
      {4096},
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0));
  data_mover.load_buffer_to_gpu_sync("blitz", t);
  data_mover.print_shm_info("blitz");
  return 0;
}