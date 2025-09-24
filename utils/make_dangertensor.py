import os
import safetensors.torch
from safetensors import safe_open
import torch
import struct
import json
import re
from typing import List

tensor_vec = []


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


def dump_tensor(tensor, tensor_name):
    if "embed" not in tensor_name:
        return
    try:
        if not tensor.is_contiguous():
            uint8_tensor = tensor.contiguous().view(torch.uint8)
        else:
            uint8_tensor = tensor.view(torch.uint8)
    except Exception:
        # not contiguous, using clone
        uint8_tensor = tensor.clone().view(torch.uint8)
    np_array = uint8_tensor.numpy()
    bytes_data = np_array.tobytes()
    with open(f"/tmp/vllm/danger-{tensor_name}.bin", "wb") as f:
        f.write(bytes_data)
        f.close()


def process_and_write_tensors(directory, output_path, safetensor_files, tp_size: int):
    """
    读取指定目录下所有 safetensors 文件，转置所有 2D 张量，并按指定顺序写入二进制文件。

    Args:
        directory (str): 包含 safetensors 文件的目录。
        output_file (str): 输出二进制文件的路径。
        tensor_order (list): 张量名称的顺序，用于写入文件。
    """

    all_tensors = {}
    all_tensors_name = []
    config_file = os.path.join(directory, "config.json")
    with open(config_file, "r") as f:
        config = json.load(f)
    vocab_size = config["vocab_size"]
    hidden_size = config["hidden_size"]
    inter_size = config["intermediate_size"]
    head_dim = (
        config["hidden_size"] // config["num_attention_heads"]
        if not "head_dim" in config.keys()
        else config["head_dim"]
    )
    num_qo_head = config["num_attention_heads"]
    num_kv_head = config["num_key_value_heads"]

    for filename in safetensor_files:
        try:
            with safe_open(filename, framework="pt") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    all_tensors[name]= tensor  # 将所有张量添加到字典中
                    all_tensors_name.append(name)
            # tensors = safetensors.torch.load_file(filename)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return

    for name in all_tensors_name:
        print(name)

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
                is_qkv_or_gateup = False
                print(f"[INIT] Name: {tensor_name}, shape: {tensor.shape}")

                # layer_id = extract_layer_id(tensor_name)
                if len(tensor.shape) == 2:
                    # NOTE: fuse gate & up proj
                    if "gate_proj" in tensor_name:
                        # cat an push to tensors
                        is_qkv_or_gateup = True
                        up_tensor_name = tensor_name.replace("gate_proj", "up_proj")
                        gate_tensor = tensor
                        up_tensor = all_tensors[up_tensor_name]
                        gate_tensor_slice = gate_tensor[
                            (tp_rank * (inter_size // tp_size)) : (
                                (tp_rank + 1) * (inter_size // tp_size)
                            ),
                            :,
                        ]
                        up_tensor_slice = up_tensor[
                            (tp_rank * (inter_size // tp_size)) : (
                                (tp_rank + 1) * (inter_size // tp_size)
                            ),
                            :,
                        ]
                        result = torch.cat([gate_tensor_slice, up_tensor_slice])
                        result = result.contiguous()
                        tensors.append((up_tensor_name, result))
                    elif "up_proj" in tensor_name:
                        is_qkv_or_gateup = True
                    # down_proj
                    elif "down_proj" in tensor_name:
                        tensor_slice = tensor[
                            :,
                            (tp_rank * (inter_size // tp_size)) : (
                                (tp_rank + 1) * (inter_size // tp_size)
                            ),
                        ]
                        # tensor_slice = torch.transpose(tensor_slice, 0, 1)
                        tensor_slice = tensor_slice.contiguous()
                        tensors.append((tensor_name, tensor_slice))
                    else:
                        if "lm_head" in tensor_name:
                            # [vocab_size, hidden_size]
                            assert (vocab_size, hidden_size) == tensor.shape
                            assert vocab_size % tp_size == 0
                            tensor_slice = tensor[
                                (tp_rank * (vocab_size // tp_size)) : (
                                    (tp_rank + 1) * (vocab_size // tp_size)
                                ),
                                :,
                            ]
                            # tensor_slice = torch.transpose(tensor_slice, 0, 1)
                            tensor_slice = tensor_slice.contiguous()
                            tensors.append((tensor_name, tensor_slice))
                        elif "embed" in tensor_name:
                            # [vocab_size, hidden_size]
                            assert (vocab_size, hidden_size) == tensor.shape
                            assert vocab_size % tp_size == 0
                            tensor_slice = tensor[
                                (tp_rank * (vocab_size // tp_size)) : (
                                    (tp_rank + 1) * (vocab_size // tp_size)
                                ),
                                :,
                            ]
                            tensor_slice = tensor_slice.contiguous()
                            tensors.append((tensor_name, tensor_slice))
                        else:
                            assert "self_attn" in tensor_name
                            assert num_qo_head % tp_size == 0
                            assert num_kv_head % tp_size == 0

                            if "o_proj" in tensor_name:  # Mistral-24B
                                # [hidden_size, num_qo_head * head_size]
                                slice_head_num = num_qo_head // tp_size
                                start_head_idx = tp_rank * slice_head_num
                                tensor_slice = tensor[
                                    :,
                                    start_head_idx
                                    * head_dim : (start_head_idx + slice_head_num)
                                    * head_dim,
                                ]
                            elif "q_proj" in tensor_name:
                                is_qkv_or_gateup = True
                                # [num_qo_head * head_size, hidden_size]
                                slice_head_num = num_qo_head // tp_size
                                start_head_idx = tp_rank * slice_head_num
                                tensor_slice = tensor[
                                    start_head_idx
                                    * head_dim : (start_head_idx + slice_head_num)
                                    * head_dim,
                                    :,
                                ]

                                k_tensor = all_tensors[
                                    tensor_name.replace("q_proj", "k_proj")
                                ]
                                v_tensor = all_tensors[
                                    tensor_name.replace("q_proj", "v_proj")
                                ]
                                k_tensor_slice = k_tensor[
                                    start_head_idx
                                    * head_dim : (start_head_idx + slice_head_num)
                                    * head_dim,
                                    :,
                                ]
                                v_tensor_slice = v_tensor[
                                    start_head_idx
                                    * head_dim : (start_head_idx + slice_head_num)
                                    * head_dim,
                                    :,
                                ]
                                result = torch.cat(
                                    [tensor_slice, k_tensor_slice, v_tensor_slice],
                                    dim=0,
                                )
                                tensors.append(
                                    (tensor_name.replace("q_proj", "v_proj"), result)
                                )
                            else:
                                is_qkv_or_gateup = True
                            # tensor_slice = torch.transpose(tensor_slice, 0, 1)
                            if is_qkv_or_gateup == False:
                                tensor_slice = tensor_slice.contiguous()
                                tensors.append((tensor_name, tensor_slice))

                        print("========================")
                else:
                    # 1-D tensor
                    tensors.append((tensor_name, tensor))
                    assert len(tensor.shape) == 1
                # 将张量转换为 NumPy 数组，以便更轻松地处理数据类型
                for name, tensor in tensors:
                    print(
                        f"Tensor {name} ele_size {tensor.element_size()} n {tensor.nelement()}"
                    )
                    tensor_vec.append(
                        (name, tensor.element_size() * tensor.nelement())
                    )
                    # print(
                    #     f"Name: {tensor_name}, dtype: {tensor.dtype}, shape: {tensor.shape}"
                    # )
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
                    # if name == "model.layers.0.mlp.down_proj.weight":
                    #     with open(f"/tmp/vllm/danger-{name}.bin", "wb") as g:
                    #         g.write(bytes_data)
                print("==========================")
                # print(tensor_map)
            with open(meta_file, "w") as ff:
                ff.write(f"{len(tensor_vec)}\n")
                for name, size in tensor_vec:
                    ff.write(f"{name} {size}\n")
                #     # print(bytes_data)

                # for tensor in tensors:

                # 写入张量数据
                # tensor_bytes = tensor_np.tobytes()
                # f.write(tensor_bytes)


if __name__ == "__main__":
    # NOTE README:
    # change `model_name` to model path
    # XXX don't forget to change num_layer (80, 40, 32, etc.)
    # XXX don't forget to change tp_size (1, 2, 4, etc.)
    # model_name = 'Llama-2-7b-chat-hf'
    # model_name = 'DeepSeek-R1-Distill-Llama-8B'
    # model_name = "Mistral-Small-24B-Instruct-2501"
    # model_name = "Qwen3-8B"
    model_name = "Qwen3-32B"
    model_directory = "/nvme/ly/models/{}".format(model_name)
    output_path = "/nvme/ly/tmp_files2"
    # tensor_order = generate_tensor_order(32)
    tp_size = 1
    process_and_write_tensors(
        model_directory, output_path, get_safetensor_files(model_directory), tp_size
    )
