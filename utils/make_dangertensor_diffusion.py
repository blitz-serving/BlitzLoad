import os
from safetensors import safe_open
import torch
import re
import argparse


def process_and_write_tensors(models_path):
    def recursive_walk_through_directory(directory: str):
        file_names = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".dangertensors"):
                    file_path = os.path.join(root, file)
                    meta_path = file_path.replace(".dangertensors", ".meta")
                    if os.path.exists(meta_path):
                        file_names.append(file_path)
        return file_names


    filenames = recursive_walk_through_directory(models_path)

    for filename in filenames:
        tensor_vec = []

        with safe_open(filename, framework="pt") as f:
            meta = f.metadata()

            output_path = filename.replace(".safetensors", ".dangertensors")
            output_meta = filename.replace(".safetensors", ".meta")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as ff:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    tensor_vec.append(
                        (
                            name,
                            tensor.element_size() * tensor.nelement(),
                            tensor.shape,
                            str(tensor.dtype),
                        )
                    )
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
            for name, size, shape, dtype in tensor_vec:
                # delete space in shape
                shape = re.sub(r"\s+", "", str(shape))
                fff.write(f"{name} {size} {shape} {dtype}\n")
            fff.write(str(meta))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-path", type=str, required=True, help="path to the models directory"
    )
    args = parser.parse_args()
    models_path = args.models_path
    process_and_write_tensors(models_path)
