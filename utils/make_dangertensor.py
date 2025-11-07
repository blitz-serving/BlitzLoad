import os
import safetensors.torch
from safetensors import safe_open
import torch
import struct
import json
import re
from typing import List
from .models import qwen2, qwen2_5_vl, qwen3
import argparse

qkv_stacks = ["q", "k", "v"]
tp_params_dim0 = ["embed", "lm_head", "qkv_proj", "gate_up_proj"]
tp_params_dim1 = ["o_proj", "down_proj"]


def get_safetensor_files(dir):
    files = [f for f in os.listdir(dir) if f.endswith(".safetensors")]
    files.sort(
        key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", x)]
    )
    return [os.path.join(dir, f) for f in files]


def extract_layer_id(tensor_name: str):
    match = re.search(r"(?:layers|h)\.(\d+)", tensor_name)
    if match:
        return int(match.group(1))
    else:
        raise "Cannot extract layer id"


def process_and_write_tensors(
    directory,
    output_path,
    safetensor_files,
    tp_size: int,
    stacked_params_mapping: List[tuple],
):
    tensor_vec = []
    all_tensors = {}
    all_tensors_name = []
    # create output_path if not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # if outputpath != directory, copy all *.json and *.txt files to output_path
    if output_path != directory:
        for file in os.listdir(directory):
            if file.endswith(".json") or file.endswith(".txt"):
                src_file = os.path.join(directory, file)
                dst_file = os.path.join(output_path, file)
                if not os.path.exists(dst_file):
                    os.system(f"cp {src_file} {dst_file}")
    config_file = os.path.join(directory, "config.json")
    with open(config_file, "r") as f:
        config = json.load(f)

    for filename in safetensor_files:
        try:
            with safe_open(filename, framework="pt") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    all_tensors[name] = tensor  # 将所有张量添加到字典中
                    all_tensors_name.append(name)
            # tensors = safetensors.torch.load_file(filename)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return

    # for name in all_tensors_name:
    #     print(name)

    # return

    for tp_rank in range(tp_size):
        filename = "dangertensors.{}.bin".format(tp_rank)
        metaname = "dangertensors.{}.meta".format(tp_rank)
        output_file = os.path.join(output_path, filename)
        meta_file = os.path.join(output_path, metaname)
        with open(output_file, "wb") as f:
            tensor_order_it = iter(all_tensors_name)
            for tensor_name in tensor_order_it:
                tensor = all_tensors[tensor_name]
                tensors = []
                is_stacked_param = False
                print(f"[INIT] Name: {tensor_name}, shape: {tensor.shape}")

                # check if name matches stacked params
                for param_name, shard_name, shard_id in stacked_params_mapping:
                    if shard_name in tensor_name:
                        is_stacked_param = True
                        shard_id = str(shard_id)
                        # whether it is qkv or gate_up_proj or something else
                        if shard_id == "q":
                            # is qkv
                            q_tensor_slice = tensor.chunk(tp_size, dim=0)[tp_rank]
                            k_tensor_slice = all_tensors[
                                tensor_name.replace("q_proj", "k_proj")
                            ].chunk(tp_size, dim=0)[tp_rank]
                            v_tensor_slice = all_tensors[
                                tensor_name.replace("q_proj", "v_proj")
                            ].chunk(tp_size, dim=0)[tp_rank]
                            result = torch.cat(
                                [q_tensor_slice, k_tensor_slice, v_tensor_slice], dim=0
                            )
                            result = result.contiguous()
                            tensors.append(
                                (tensor_name.replace("q_proj", "v_proj"), result)
                            )
                        elif shard_id == "0":
                            # gate_up or something else
                            # get all slices taht share one param_name
                            tensor_slices = []
                            last_name = ""
                            for pn, sn, sid in stacked_params_mapping:
                                if pn == param_name:
                                    t = all_tensors[tensor_name.replace(shard_name, sn)]
                                    tensor_slices.append(
                                        t.chunk(tp_size, dim=0)[tp_rank]
                                    )
                                    last_name = sn
                            # sort tensor_slices by shard_id
                            tensor_slices = sorted(
                                tensor_slices,
                                key=lambda x: [
                                    int(t) if t.isdigit() else t
                                    for t in re.split(r"(\d+)", sn)
                                ],
                            )
                            # concat all slices
                            result = torch.cat(tensor_slices, dim=0)
                            result = result.contiguous()
                            tensors.append(
                                (tensor_name.replace(shard_name, last_name), result)
                            )

                if is_stacked_param is False:
                    if any(x in tensor_name for x in tp_params_dim0):
                        # should slice
                        tensor_slice = tensor.chunk(tp_size, dim=0)[tp_rank]
                        tensors.append((tensor_name, tensor_slice))
                    elif any(x in tensor_name for x in tp_params_dim1):
                        if tensor.dim() == 1:
                            tensor_slice = tensor.chunk(tp_size, dim=0)[tp_rank]
                        else:
                            tensor_slice = tensor.chunk(tp_size, dim=1)[tp_rank]
                        tensors.append((tensor_name, tensor_slice))
                    else:
                        # no need to slice
                        tensors.append((tensor_name, tensor))

                for name, tensor in tensors:
                    print(
                        f"Tensor {name} ele_size {tensor.element_size()} n {tensor.nelement()}"
                    )
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
                        # print(f"Tensor stride: {tensor.stride()}")
                        if not tensor.is_contiguous():
                            tensor = tensor.contiguous()
                            uint8_tensor = tensor.view(torch.uint8)
                    np_array = uint8_tensor.numpy()
                    bytes_data = np_array.tobytes()
                    f.write(bytes_data)
                print("==========================")
            with open(meta_file, "w") as ff:
                ff.write(f"{len(tensor_vec)}\n")
                for name, size, shape, dtype in tensor_vec:
                    shape = re.sub(r"\s+", "", str(shape))
                    ff.write(f"{name} {size} {shape} {dtype}\n")
                tensor_vec = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="model-path")
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="output-path, can be equal to model-path",
    )
    parser.add_argument(
        "--tp-size", type=int, required=True, help="tensor parallel size"
    )
    args = parser.parse_args()
    model_path = args.model_path
    output_path = args.output_path
    tp_size = args.tp_size

    stacked_params_mapping = qwen2.stacked_param_mapping
    process_and_write_tensors(
        model_path,
        output_path,
        get_safetensor_files(model_path),
        tp_size,
        stacked_params_mapping,
    )
