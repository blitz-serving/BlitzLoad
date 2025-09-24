import grpc
import torch
import ctypes
import time
import cProfile, pstats, io

import functools
import inspect
from . import generate_pb2
from . import generate_pb2_grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

import os
from threading import Lock

_grpc_clients: dict[str, dict] = {}
_grpc_clients_lock = Lock()

_rank_info: dict[str, int] = {}  # proc_id: rank
profiler = cProfile.Profile()

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

for var in proxy_vars:
    if var in os.environ:
        del os.environ[var]


CUDA_IPC_HANDLE_SIZE = 64
TASK_ID = ""
LIB_TIME = 0

try:
    import cupy.cuda.runtime as rt

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def health_check(channel) -> bool:
    stub = health_pb2_grpc.HealthStub(channel)
    try:
        response = stub.Check(health_pb2.HealthCheckRequest(service=""))
        if response.status == health_pb2.HealthCheckResponse.SERVING:
            return True
        else:
            return False
    except Exception as e:
        return False


def register_fork_handler(channel):
    def _fork_post_child():
        channel.close()
        grpc._cygrpc.terminate()
        print("Child process: gRPC resources destroyed")

    os.register_at_fork(after_in_child=_fork_post_child)


def register_rank(rank: int):
    proc_id = os.getpid()
    _rank_info[proc_id] = rank
    print(f"Proc {proc_id}'s rank is {rank}")


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
        self,
        device_ptr: int,
        tensor: torch.Tensor,
        size: int,
        tensor_offset: int = 0,
    ):
        if not tensor.is_cuda:
            raise ValueError("Target tensor must be on CUDA device")

        if not tensor.is_contiguous():
            raise MemoryError("Tensor should be contiguous")
        tensor_ptr = tensor.data_ptr() + tensor_offset

        rt.memcpy(tensor_ptr, device_ptr, size, rt.memcpyDeviceToDevice)


class cudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * CUDA_IPC_HANDLE_SIZE)]


cuda_mem_manager = CudaMemManager()


def _get_stub(server_addr: str, refresh_stub: bool = False):
    global _grpc_clients

    current_pid = os.getpid()

    with _grpc_clients_lock:
        if current_pid not in _grpc_clients:
            _grpc_clients[current_pid] = {
                "checked": False,
                "channel": None,
                "stub": None,
            }

        client_info = _grpc_clients[current_pid]

        if (
            client_info["checked"] != True
            or refresh_stub
            or client_info["stub"] is None
        ):
            print(f"[PID {current_pid}] Creating new stub for {server_addr}")

            channel = grpc.insecure_channel(server_addr)
            if not health_check(channel):
                register_fork_handler(channel)

                channel = grpc.insecure_channel(server_addr)

            try:
                grpc.channel_ready_future(channel).result(timeout=5)
                print(f"[PID {current_pid}] Connected to {server_addr}")
            except grpc.FutureTimeoutError:
                raise RuntimeError(f"Cannot connect to {server_addr}")
            stub = generate_pb2_grpc.ParamServiceStub(channel)

            client_info["checked"] = True
            client_info["channel"] = channel
            client_info["stub"] = stub

    return client_info["stub"]


def load_weight_from_ipc_handle(
    param: torch.Tensor, weight_name: str, server_addr: str = "localhost:60060"
) -> None:
    global LIB_TIME
    rank = _rank_info[os.getpid()]
    # profiler.enable()
    start = time.time()
    stub = _get_stub(server_addr)
    tensor_size = param.element_size() * param.nelement()
    req = generate_pb2.GetHandlerRequest(
        tensor_name=weight_name, tensor_size=tensor_size, rank=rank
    )
    try:
        # start = time.time()
        res = stub.GetHandler(req, timeout=5.0)
        # H2D_TIME += time.time()-start
    except grpc.RpcError as e:
        print("GetHandler RPC failed:", e.code(), e.details(), flush=True)
        return

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
        tensor_name=weight_name, tensor_size=loaded_bytes, rank=rank
    )
    stub.RevertHandler(req)

    if loaded_bytes < tensor_size:
        while loaded_bytes < tensor_size:
            # print(f"continue loading {weight_name}")
            req = generate_pb2.GetHandlerRequest(
                tensor_name=weight_name,
                tensor_size=tensor_size - loaded_bytes,
                rank=rank,
            )
            # start = time.time()
            res = stub.GetHandler(req)
            # H2D_TIME += time.time() - start
            handle_bytes = res.ipc_handler
            new_loaded_bytes = res.loaded_size
            device_offset = res.offset
            if len(handle_bytes) != CUDA_IPC_HANDLE_SIZE:
                raise ValueError(
                    f"Invalid IPC handle size: expected {CUDA_IPC_HANDLE_SIZE}, got {len(handle_bytes)}"
                )
            device_ptr = (
                cuda_mem_manager.cuda_ipc_handle_to_ptr(handle_bytes) + device_offset
            )
            cuda_mem_manager.copy_device_to_tensor(
                device_ptr, param, new_loaded_bytes, loaded_bytes
            )

            loaded_bytes += new_loaded_bytes
            req = generate_pb2.RevertHandlerRequest(
                tensor_name=weight_name, tensor_size=new_loaded_bytes, rank=rank
            )
            stub.RevertHandler(req)
            # print(f"Current H2D time: {H2D_TIME}s")
    else:
        pass
        # print(f"Current H2D time: {H2D_TIME}s")
        # print(f"tensor {weight_name} load done")

    LIB_TIME += time.time() - start
    # print(f"Current Lib time: {LIB_TIME}s")
    # profiler.disable()


def pull_model(model_name: str, world_size: int, server_addr: str = "localhost:60060"):
    global TASK_ID, S2H_TIME
    stub = _get_stub(server_addr)
    # S2H_TIME = time.time()
    req = generate_pb2.PullModelRequest(model_name=model_name, world_size=world_size)
    res = stub.PullModel(req)
    pid = os.getpid()
    _grpc_clients[pid]["channel"].close()
    TASK_ID = res.task_id
    return TASK_ID


def check_model(server_addr="localhost:60060") -> bool:
    # global S2H_TIME
    stub = _get_stub(server_addr)
    req = generate_pb2.CheckModelRequest(task_id=TASK_ID)
    res = stub.CheckModel(req)
    # if res.done:
    # S2H_TIME = time.time() - S2H_TIME
    # print(f"Load model time: {S2H_TIME}, bandwidth: {16 / S2H_TIME}GBps")
    return res.done


def print_profile():
    pass
    # stats = pstats.Stats(profiler).sort_stats("cumtime")
    # stats.print_stats()


def _dump_tensor(tensor, tensor_name, out_dir):
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


# FIXME: gguf not applied
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
