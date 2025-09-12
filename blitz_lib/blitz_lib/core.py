import grpc
import torch
import ctypes

from . import generate_pb2
from . import generate_pb2_grpc

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


CUDA_IPC_HANDLE_SIZE = 64
libcuda = ctypes.CDLL("libcuda.so")


class cudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * CUDA_IPC_HANDLE_SIZE)]


_channel = None
_stub = None


def _get_stub(server_addr: str):
    global _channel, _stub
    if _stub is not None:
        return _stub

    _channel = grpc.insecure_channel(server_addr)
    try:
        grpc.channel_ready_future(_channel).result(timeout=5)
        print(f"[blitz_param_engine_cli] Connected to {server_addr}")
    except grpc.FutureTimeoutError:
        raise RuntimeError(f"Cannot connect to {server_addr}")

    _stub = generate_pb2_grpc.ParamServiceStub(_channel)
    return _stub


def send_grpc_call_to_server(
    param: torch.Tensor, weight_name: str, server_addr: str = "localhost:60060"
) -> None:
    if not param.is_cuda:
        raise ValueError("param must on cuda device")

    tensor_size = param.numel()
    dtype_size = param.element_size()
    device = param.device.index
    nbytes = tensor_size * dtype_size
    ptr = param.data_ptr()

    handle = cudaIpcMemHandle()
    res = libcuda.cuIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(ptr))
    if res != 0:
        raise RuntimeError(f"cuIpcGetMemHandle failed: {res}")

    handle_bytes = ctypes.string_at(ctypes.byref(handle), CUDA_IPC_HANDLE_SIZE)

    req = generate_pb2.LoadWeightRequest(
        ipc_handle=handle_bytes,
        tensor_size=nbytes,
        tensor_device=device,
        tensor_name=weight_name,
    )

    stub = _get_stub(server_addr)
    return stub.LoadWeight(req)


def pull_model(model_name: str, server_addr: str = "localhost:60060"):
    stub = _get_stub(server_addr)
    req = generate_pb2.PullModelRequest(model_name=model_name)
    return stub.PullModel(req)
