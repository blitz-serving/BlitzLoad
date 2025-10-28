import os
import safetensors.torch
from safetensors import safe_open
import torch
import struct
import json
import re
from typing import List
import argparse


filenames = [
    "/nvme/ly/ComfyUI/models/vae/ae.safetensors",
    "/nvme/ly/ComfyUI/models/diffusion_models/flux1-schnell.safetensors",
    "/nvme/ly/ComfyUI/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors",
    "/nvme/ly/ComfyUI/models/text_encoders/clip_l.safetensors",
]
for filename in filenames:
    all_tensors = {}
    all_tensors_name = []
    tensor_vec = []

    with safe_open(filename, framework="pt") as f:
        output_path = filename.replace(".safetensors", ".dangertensors")
        output_meta = output_path + ".meta"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as ff:
            for name in f.keys():
                tensor = f.get_tensor(name)
                tensor_vec.append((name, tensor.element_size() * tensor.nelement()))
                try:
                    uint8_tensor = tensor.view(torch.uint8)
                except Exception:
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                        uint8_tensor = tensor.view(torch.uint8)
                np_array = uint8_tensor.numpy()
                bytes_data = np_array.tobytes()
                ff.write(bytes_data)
    with open(output_meta, "w") as fff:
        fff.write(f"{len(tensor_vec)}\n")
        for name, size in tensor_vec:
            fff.write(f"{name} {size}\n")

# for name in all_tensors_name:
#     tensor = all_tensors[name]
#     shape = tensor.shape
#     dtype = tensor.dtype
#     print(f"Name: {name}, Shape: {shape}, Dtype: {dtype}")