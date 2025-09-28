import grpc
import torch
import ctypes
import time
import cProfile, pstats, io

import json
import functools
import inspect
import zmq
from . import generate_pb2
from . import generate_pb2_grpc
from . import mq_types
from grpc_health.v1 import health_pb2, health_pb2_grpc

import os
from threading import Lock


class MqInfo:
    def __init__(self, context, port_dict):
        self.context = context
        self.port_dict = port_dict


_zmq_sockets: dict[int, MqInfo] = {}
_zmq_clients_lock = Lock()

_rank_info: dict[str, int] = {}  # proc_id: rank
profiler = cProfile.Profile()
# context = zmq.Context()

server_addr_map = {
    "pull_model": "tcp://localhost:55555",
    "check_model": "tcp://localhost:55556",
    "load_param": "tcp://localhost:55557",
    "revert_handler": "tcp://localhost:55558",
    "reset_status": "tcp://localhost:55559",
}

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

        # print(f"Tensor ptr {tensor_ptr}, device ptr {device_ptr}, size {size}")

        rt.memcpy(tensor_ptr, device_ptr, size, rt.memcpyDeviceToDevice)


class cudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * CUDA_IPC_HANDLE_SIZE)]


cuda_mem_manager = CudaMemManager()


def _get_socket(server_addr: str, refresh_socket: bool = False):
    global _zmq_sockets
    current_pid = os.getpid()

    with _zmq_clients_lock:
        if current_pid not in _zmq_sockets:
            _zmq_sockets[current_pid] = MqInfo(context=zmq.Context(), port_dict={})

        cli_info = _zmq_sockets[current_pid]

        if refresh_socket or server_addr not in cli_info.port_dict:
            print(f"[PID {current_pid}] Creating new socket for {server_addr}")
            socket = cli_info.context.socket(zmq.REQ)
            socket.connect(server_addr)

            cli_info.port_dict[server_addr] = socket

    return cli_info.port_dict[server_addr]


def _send_recv(socket, object):
    req_json = json.dumps(object.__dict__)
    socket.send_string(req_json)
    reply = socket.recv_string()
    resp_dict = json.loads(reply)
    return resp_dict


def load_tensor(param: torch.Tensor, weight_name: str):
    loaded_bytes = 0
    socket_load = _get_socket(server_addr_map["load_param"])
    socket_revert = _get_socket(server_addr_map["revert_handler"])
    rank = _rank_info[os.getpid()]
    tensor_size = param.element_size() * param.nelement()
    while loaded_bytes < tensor_size:
        req = mq_types.LoadTensorRequest(
            tensor_name=weight_name, tensor_size=tensor_size - loaded_bytes, rank=rank
        )
        resp_dict = _send_recv(socket_load, req)
        resp = mq_types.LoadTensorResponse(
            resp_dict["handler"], resp_dict["offset"], resp_dict["loaded_size"]
        )
        device_ptr = (
            cuda_mem_manager.cuda_ipc_handle_to_ptr(bytes(resp.handler)) + resp.offset
        )
        cuda_mem_manager.copy_device_to_tensor(
            device_ptr, param, resp.loaded_size, loaded_bytes
        )
        loaded_bytes += resp.loaded_size

        req = mq_types.RevertHandlerRequest(weight_name, resp.loaded_size, rank)
        _send_recv(socket_revert, req)


def pull_model(model_name: str, world_size: int, tp_size: int, pp_size: int):
    global TASK_ID
    socket = _get_socket(server_addr=server_addr_map["pull_model"])
    req = mq_types.PullModelRequest(
        model_name=model_name, world_size=world_size, tp_size=tp_size, pp_size=pp_size
    )
    resp_dict = _send_recv(socket, req)
    TASK_ID = resp_dict["task_id"]
    return TASK_ID


def check_model(model_name: str) -> bool:
    # global S2H_TIME
    socket = _get_socket(server_addr=server_addr_map["check_model"])
    req = mq_types.CheckModelRequest(model_name=model_name, task_id=TASK_ID)
    resp_dict = _send_recv(socket, req)
    return resp_dict["done"]


def reset_status():
    socket = _get_socket(server_addr=server_addr_map["reset_status"])
    rank = _rank_info[os.getpid()]
    req = mq_types.ResetStatusRequest(rank=rank)
    print("Send reset")
    _send_recv(socket, req)


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

        load_tensor(val, bound.arguments[param_names[0]].prefix)
        # _dump_tensor(val, bound.arguments[param_names[0]].prefix)

    return wrapper
