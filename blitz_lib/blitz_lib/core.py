import grpc
import torch
import ctypes

import functools
import inspect
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

try:
    import cupy.cuda.runtime as rt

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class CudaMemManager:
    def __init__(self):
        if not HAS_CUPY:
            raise ImportError("pycuda is required for CudaMemManager")

    def cuda_ipc_handle_to_ptr(self, ipc_handle: bytes) -> int:
        cuda_ipc_handle = 64
        if len(ipc_handle) != cuda_ipc_handle:
            raise ValueError(
                f"Invalid IPC handle size: expected {cuda_ipc_handle}, got {len(ipc_handle)}"
            )

        try:
            device_ptr = rt.ipcOpenMemHandle(
                ipc_handle, rt.cudaIpcMemLazyEnablePeerAccess
            )
            return device_ptr
        except Exception as e:
            raise RuntimeError(f"Failed to open IPC memory: {e}")

    def copy_device_to_tensor(
        self, device_ptr: int, tensor: torch.Tensor, size: int, tensor_offset: int = 0,
    ):
        if not tensor.is_cuda:
            raise ValueError("Target tensor must be on CUDA device")

        if not tensor.is_contiguous():
            raise MemoryError("Tensor should be contiguous")
            # tensor = tensor.contiguous()
        tensor_ptr = tensor.data_ptr() + tensor_offset
        print(f"device_ptr={device_ptr}, tensor_ptr={tensor_ptr}, size={size}")

        rt.memcpy(tensor_ptr, device_ptr, size, rt.memcpyDeviceToDevice)


class cudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * CUDA_IPC_HANDLE_SIZE)]


_channel = None
_stub = None
cuda_mem_manager = CudaMemManager()


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


def load_weight_from_ipc_handle(
    param: torch.Tensor, weight_name: str, server_addr: str = "localhost:60060"
) -> None:
    print(f"Tensor name: {weight_name}")
    stub = _get_stub(server_addr)
    tensor_size = param.element_size() * param.nelement()
    req = generate_pb2.GetHandlerRequest(
        tensor_name=weight_name, tensor_size=tensor_size
    )
    res = stub.GetHandler(req)
    handle_bytes = res.ipc_handler
    loaded_bytes = res.loaded_size
    device_offset = res.offset
    if len(handle_bytes) != CUDA_IPC_HANDLE_SIZE:
        raise ValueError(
            f"Invalid IPC handle size: expected {CUDA_IPC_HANDLE_SIZE}, got {len(handle_bytes)}"
        )
    device_ptr = cuda_mem_manager.cuda_ipc_handle_to_ptr(handle_bytes) + device_offset
    cuda_mem_manager.copy_device_to_tensor(device_ptr, param, loaded_bytes)
    req = generate_pb2.RevertHandlerRequest(
        tensor_name=weight_name, tensor_size=loaded_bytes
    )
    stub.RevertHandler(req)

    if loaded_bytes < tensor_size:
        # tensor too large
        print(f"tensor {weight_name} too large")
        while loaded_bytes < tensor_size:
            print(f"continue loading {weight_name}")
            req = generate_pb2.GetHandlerRequest(
                tensor_name=weight_name, tensor_size=tensor_size - loaded_bytes
            )
            res = stub.GetHandler(req)
            handle_bytes = res.ipc_handler
            new_loaded_bytes = res.loaded_size
            device_offset = res.offset
            if len(handle_bytes) != CUDA_IPC_HANDLE_SIZE:
                raise ValueError(
                    f"Invalid IPC handle size: expected {CUDA_IPC_HANDLE_SIZE}, got {len(handle_bytes)}"
                )
            device_ptr = cuda_mem_manager.cuda_ipc_handle_to_ptr(handle_bytes) + device_offset
            # cuda_mem_manager.copy_device_to_tensor(
            #     device_ptr, param[int(loaded_bytes / param.element_size()):], new_loaded_bytes
            # )
            cuda_mem_manager.copy_device_to_tensor(
                device_ptr, param, new_loaded_bytes, loaded_bytes
            )

            loaded_bytes += new_loaded_bytes
            print(f"Current size: {loaded_bytes}")
            # inform engine that handler can be destroyed
            req = generate_pb2.RevertHandlerRequest(
                tensor_name=weight_name, tensor_size=new_loaded_bytes
            )
            stub.RevertHandler(req)
    else:
        print(f"tensor {weight_name} load done")


def pull_model(model_name: str, server_addr: str = "localhost:60060"):
    stub = _get_stub(server_addr)
    req = generate_pb2.PullModelRequest(model_name=model_name)
    return stub.PullModel(req)


def _dump_tensor(tensor, tensor_name, out_dir):
    # if "norm" in tensor_name:
    #     print(f"{tensor_name}: {tensor.shape}, {tensor.dtype}, {tensor.device}, first value: {tensor.view(torch.float32)[0]}")
    # return
    # if "norm" not in tensor_name:
    #     return
    tensor = tensor.cpu()
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
    with open(f"/tmp/{out_dir}/{tensor_name}.bin", "wb") as f:
        f.write(bytes_data)
        f.close()


def vllm_hook(func):
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        param_names = list(sig.parameters.keys())
        name = param_names[1]
        val = bound.arguments[name]
        load_weight_from_ipc_handle(val, bound.arguments[param_names[0]].prefix)
        # _dump_tensor(val, bound.arguments[param_names[0]].prefix)

    return wrapper


def vllm_dumper(func):
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        param_names = list(sig.parameters.keys())
        name = param_names[1]
        val = bound.arguments[name]
        return result

    return wrapper
