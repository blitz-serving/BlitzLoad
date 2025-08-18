import torch
import blitz_lib

dm = blitz_lib.blitz_lib()
dm.init(blitz_lib.Mode.CPU)
dm.register_buffer(5 * 1024 * 1024 * 1024, "blitz")

dm.load_file_to_buffer_sync(
    "blitz", ["/nvme/models/Meta-Llama-3-8B-Instruct/model-00001-of-00004.safetensors"]
)

t = torch.zeros((4096,), dtype=torch.bfloat16, device="cuda:0")
dm.load_buffer_to_gpu_sync("blitz", t)

dm.print_shm_info("blitz")
