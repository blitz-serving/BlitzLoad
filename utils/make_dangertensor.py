import os
import safetensors.torch
from safetensors import safe_open
import torch
import struct
import json
import re
from typing import List


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

def process_visual_tensor(tensor_name, tensor, tp_size, tp_rank):
    return tensor
    if "attn.qkv" in tensor_name:
        if "weight" in tensor_name:
            return tensor
        else:
            return tensor
    elif "attn.proj" in tensor_name:
        return tensor
    elif "mlp" in tensor_name and "fc1" in tensor_name:
        return tensor
    elif "deepstack_merger_list" in tensor_name:
        return tensor
    elif 'merger.' in tensor_name and 'deepstack_merger_list' not in tensor_name:
        if 'linear_fc1' in tensor_name:
            if 'weight' in tensor_name:
                return tensor.chunk(tp_size, dim=0)[tp_rank]
            else:  # bias
                return tensor.chunk(tp_size, dim=0)[tp_rank]
                
        elif 'linear_fc2' in tensor_name:
            if 'weight' in tensor_name:
                return tensor.chunk(tp_size, dim=1)[tp_rank]
            else:  # bias
                return tensor
                
        elif 'norm' in tensor_name:
            return tensor
    
    # patch embedding
    elif 'patch_embed' in tensor_name:
        if 'weight' in tensor_name:
            return tensor.chunk(tp_size, dim=0)[tp_rank]
        else:  # bias
            return tensor.chunk(tp_size, dim=0)[tp_rank]
    
    # positional embedding  
    elif 'pos_embed' in tensor_name:
        return tensor.chunk(tp_size, dim=1)[tp_rank]
    else:
        assert("norm" in tensor_name)
        # norm
        return tensor
        

def process_and_write_tensors(directory, output_path, safetensor_files, tp_size: int, is_vision_model: bool = False):
    """
    读取指定目录下所有 safetensors 文件，转置所有 2D 张量，并按指定顺序写入二进制文件。

    Args:
        directory (str): 包含 safetensors 文件的目录。
        output_file (str): 输出二进制文件的路径。
        tensor_order (list): 张量名称的顺序，用于写入文件。
    """

    tensor_vec = []
    all_tensors = {}
    all_tensors_name = []
    config_file = os.path.join(directory, "config.json")
    with open(config_file, "r") as f:
        config = json.load(f)
    if is_vision_model:
        text_config = config["text_config"]
    else:
        text_config = config
    vocab_size = text_config["vocab_size"]
    hidden_size = text_config["hidden_size"]
    inter_size = text_config["intermediate_size"]
    head_dim = (
        text_config["hidden_size"] // text_config["num_attention_heads"]
        if not "head_dim" in text_config.keys()
        else text_config["head_dim"]
    )
    num_qo_head = text_config["num_attention_heads"]
    num_kv_head = text_config["num_key_value_heads"]

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
                # if tensor_name != "model.visual.deepstack_merger_list.0.norm.bias":
                #     continue

                if "visual" in tensor_name:
                    tensor_slice = process_visual_tensor(tensor_name, tensor, tp_size, tp_rank)
                    tensors.append((tensor_name, tensor_slice))
                # layer_id = extract_layer_id(tensor_name)
                else:
                    # text
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
                            if "gate_up_proj" in tensor_name: # fused case
                                tensor_slice = tensor.chunk(tp_size, dim=0)[tp_rank]
                                tensors.append((tensor_name, tensor_slice))
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
                            elif "gate" in tensor_name and "mlp" in tensor_name:
                                # moe gate, no need to slice
                                tensors.append((tensor_name, tensor))
                                # tensors.append((tensor_name, tensor.chunk(tp_size, dim=0)[tp_rank]))
                            else:
                                assert "self_attn" in tensor_name
                                assert num_qo_head % tp_size == 0
                                assert num_kv_head % tp_size == 0

                                if "o_proj" in tensor_name:
                                    # [hidden_size, num_qo_head * head_size]
                                    slice_head_num = num_qo_head // tp_size
                                    start_head_idx = tp_rank * slice_head_num
                                    tensor_slice = tensor[
                                        :,
                                        start_head_idx
                                        * head_dim : (start_head_idx + slice_head_num)
                                        * head_dim,
                                    ]
                                elif "q_proj" in tensor_name or "k_proj" in tensor_name or "v_proj" in tensor_name:
                                    is_qkv_or_gateup = True
                                    # [num_qo_head * head_size, hidden_size]
                                    tensor_slice = tensor.chunk(tp_size, dim=0)[tp_rank]
                                    tensors.append((tensor_name, tensor_slice))
                                    # k_tensor = all_tensors[
                                    #     tensor_name.replace("q_proj", "k_proj")
                                    # ]
                                    # v_tensor = all_tensors[
                                    #     tensor_name.replace("q_proj", "v_proj")
                                    # ]
                                    # k_tensor_slice = k_tensor.chunk(tp_size, dim=0)[tp_rank]
                                    # v_tensor_slice = v_tensor.chunk(tp_size, dim=0)[tp_rank]
                                    # result = torch.cat(
                                    #     [q_tensor_slice, k_tensor_slice, v_tensor_slice], dim=0
                                    # )
                                    # result = result.contiguous()
                                    # tensors.append(
                                    #     (tensor_name.replace("q_proj", "v_proj"), result)
                                    # )
                                    # print(f"Q size: {tensor.shape} -> {q_tensor_slice.shape}")
                                    # print(f"K size: {k_tensor.shape} -> {k_tensor_slice.shape}")
                                    # print(f"V size: {v_tensor.shape} -> {v_tensor_slice.shape}")
                                    

                                else:
                                    is_qkv_or_gateup = True
                                # tensor_slice = torch.transpose(tensor_slice, 0, 1)
                                if is_qkv_or_gateup == False:
                                    tensor_slice = tensor_slice.contiguous()
                                    tensors.append((tensor_name, tensor_slice))

                            print("========================")
                    elif tensor.dim() == 3:
                        # no need to slice expert tensors
                        tensors.append((tensor_name, tensor))
                    else:
                        # 1-D tensor
                        assert len(tensor.shape) == 1
                        if "self_attn" in tensor_name and "bias" in tensor_name:
                            assert num_qo_head % tp_size == 0
                            assert num_kv_head % tp_size == 0
                            if "q_proj.bias" in tensor_name:
                                is_qkv_or_gateup = True
                                chunk_size = hidden_size // tp_size
                                tensor_slice = tensor[
                                    chunk_size * tp_rank : chunk_size * (tp_rank + 1)
                                ]
                                k_tensor = all_tensors[
                                    tensor_name.replace("q_proj", "k_proj")
                                ]
                                v_tensor = all_tensors[
                                    tensor_name.replace("q_proj", "v_proj")
                                ]
                                kv_chunk_size = hidden_size * num_kv_head // tp_size // num_qo_head 
                                k_tensor_slice = k_tensor[
                                    kv_chunk_size
                                    * tp_rank : kv_chunk_size
                                    * (tp_rank + 1)
                                ]
                                v_tensor_slice = v_tensor[
                                    kv_chunk_size
                                    * tp_rank : kv_chunk_size
                                    * (tp_rank + 1)
                                ]
                                result = torch.cat(
                                    [tensor_slice, k_tensor_slice, v_tensor_slice],
                                    dim=0,
                                )
                                tensors.append(
                                    (tensor_name.replace("q_proj", "v_proj"), result)
                                )
                        else:
                            tensors.append((tensor_name, tensor))

                for name, tensor in tensors:
                    print(
                        f"Tensor {name} ele_size {tensor.element_size()} n {tensor.nelement()}"
                    )
                    tensor_vec.append((name, tensor.element_size() * tensor.nelement()))
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
                tensor_vec = []

                # for tensor in tensors:

                # 写入张量数据
                # tensor_bytes = tensor_np.tobytes()
                # f.write(tensor_bytes)


if __name__ == "__main__":
    model_name = "Qwen3-VL-30B"
    model_directory = "/nvme/ly/models/{}".format(model_name)
    output_path = "/nvme/ly/models/Qwen3-VL-30B"
    tp_size = 2
    process_and_write_tensors(
        model_directory, output_path, get_safetensor_files(model_directory), tp_size, "vl" in model_name.lower()
    )
