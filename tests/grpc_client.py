import grpc
import torch
import cupy as cp
import generate_pb2
import generate_pb2_grpc
import ctypes


# 导入 CUDA Driver API
libcuda = ctypes.CDLL("libcuda.so")


class cudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]  # cudaIpcMemHandle_t 64 bytes


import os

# 需要清除的代理环境变量列表
proxy_vars = [
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
]

# 删除它们
for var in proxy_vars:
    if var in os.environ:
        del os.environ[var]

print("All proxy environment variables have been unset.")


def test_param_service(server_addr="localhost:60060"):
    # 建立 gRPC 连接
    channel = grpc.insecure_channel(server_addr)

    try:
        # 等待最多 5 秒，检查 channel 是否 ready
        grpc.channel_ready_future(channel).result(timeout=5)
        print(f"Successfully connected to {server_addr}")
    except grpc.FutureTimeoutError:
        raise f"Cannot connect to {server_addr}"

    stub = generate_pb2_grpc.ParamServiceStub(channel)
    
    # 测试 PullModel 接口
    model_name = "/nvme/ly/tmp_files"
    print(f"Calling PullModel with model_name='{model_name}'...")
    pull_req = generate_pb2.PullModelRequest(model_name=model_name)
    try:
        pull_resp = stub.PullModel(pull_req)
    except grpc.RpcError as e:
        print("PullModel failed:", e)

    # 测试 LoadWeight 接口

    tensor_size = 16777216  # 假设 tensor 的元素数量
    dtype_size = 2  # bf16
    tensor_device = 4
    nbytes = tensor_size * dtype_size
    x = torch.zeros(tensor_size, dtype=torch.bfloat16, device="cuda:4")
    ptr = x.data_ptr()

    handle = cudaIpcMemHandle()
    res = libcuda.cuIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(ptr))
    if res != 0:
        raise RuntimeError(f"cuIpcGetMemHandle failed: {res}")

    # handle 可以用 ctypes.string_at(handle, 64) 转成 bytes
    handle_bytes = ctypes.string_at(ctypes.byref(handle), 64)
    print("send")
    load_req = generate_pb2.LoadWeightRequest(
        ipc_handle=handle_bytes,
        tensor_size=nbytes,
        tensor_device=tensor_device,
        tensor_name="model.layers.0.self_attn.q_proj.weight",
    )

    try:
        load_resp = stub.LoadWeight(load_req)
        print("LoadWeight response:", load_resp)
    except grpc.RpcError as e:
        print("LoadWeight failed:", e)
    print(x[0:5])


if __name__ == "__main__":
    test_param_service()
