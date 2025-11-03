import grpc
import torch
import ctypes
import time
import cProfile, pstats, io

import json
import functools
import inspect
import zmq
from multiprocessing import Manager, Process
import ast
from . import mq_types

from multiprocessing import shared_memory

import os
from threading import Lock


CUDA_IPC_HANDLE_SIZE = 64
LIB_TIME = 0
PORT = 35555
SHM_NAME = "task_id_test"
HEADER_SIZE = 8


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
    "pull_model": f"tcp://localhost:{PORT}",
    "check_model": f"tcp://localhost:{PORT + 1}",
    "load_param": f"tcp://localhost:{PORT + 2}",
    "revert_handler": f"tcp://localhost:{PORT + 3}",
    "reset_status": f"tcp://localhost:{PORT + 4}",
    "pull_diffusion_model": f"tcp://localhost:{PORT + 5}",
    "load_meta": f"tcp://localhost:{PORT + 6}",
    "load_tensor_meta": f"tcp://localhost:{PORT + 7}",
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


try:
    import cupy.cuda.runtime as rt

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


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


def get_rank():
    proc_id = os.getpid()
    return _rank_info[proc_id]


class CudaMemManager:
    def __init__(self):
        if not HAS_CUPY:
            raise ImportError("CUPY is required for CudaMemManager")

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
            print(f"[PID {current_pid}] Creating new socket for {server_addr} done")

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
    # rank = _rank_info[os.getpid()]
    tensor_size = param.element_size() * param.nelement()
    while loaded_bytes < tensor_size:
        req = mq_types.LoadTensorRequest(
            tensor_name=weight_name, tensor_size=tensor_size - loaded_bytes, rank=0
        )
        resp_dict = _send_recv(socket_load, req)
        print("Receive load tensor response")
        resp = mq_types.LoadTensorResponse(
            resp_dict["handler"],
            resp_dict["offset"],
            resp_dict["loaded_size"],
            resp_dict["resize_tensor"],
        )
        device_ptr = (
            cuda_mem_manager.cuda_ipc_handle_to_ptr(bytes(resp.handler)) + resp.offset
        )
        if resp.resize_tensor:
            print(f"Tensor size {tensor_size} != loaded size{resp.loaded_size}")
            tensor_size = resp.loaded_size
            assert False
        cuda_mem_manager.copy_device_to_tensor(
            device_ptr, param, resp.loaded_size, loaded_bytes
        )
        loaded_bytes += resp.loaded_size

        req = mq_types.RevertHandlerRequest(weight_name, resp.loaded_size, 0)
        _send_recv(socket_revert, req)


def pull_model(model_name: str, world_size: int, tp_size: int, pp_size: int):
    socket = _get_socket(server_addr=server_addr_map["pull_model"])
    req = mq_types.PullModelRequest(
        model_name=model_name, world_size=world_size, tp_size=tp_size, pp_size=pp_size
    )
    resp_dict = _send_recv(socket, req)
    task_id = resp_dict["task_id"]
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=1024 * 1024)
    d = {"model_name": task_id}
    b = json.dumps(d).encode("utf-8")
    payload_size = len(b)
    shm.buf[:HEADER_SIZE] = payload_size.to_bytes(HEADER_SIZE, byteorder="little")
    shm.buf.cast('B')[int(HEADER_SIZE) : int(HEADER_SIZE) + len(b)] = bytes(b)
    return task_id


def pull_diffusion_model(directory: str):
    # walk through the directory to get all file names
    file_names = recursive_walk_through_directory(directory)

    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=1024 * 1024)
    d = {}
    # print(file_names)
    for file_name in file_names:
        print(f"Pull diffusion model file: {file_name}")
        req = mq_types.PullDiffusionModelRequest(file_name=file_name)
        resp_dict = _send_recv(
            socket=_get_socket(server_addr=server_addr_map["pull_diffusion_model"]),
            object=req,
        )
        task_id = resp_dict["task_id"]
        d[file_name] = task_id

    b = json.dumps(d).encode("utf-8")
    payload_size = len(b)
    shm.buf[:HEADER_SIZE] = payload_size.to_bytes(HEADER_SIZE, byteorder="little")
    shm.buf.cast('B')[int(HEADER_SIZE) : int(HEADER_SIZE) + len(b)] = bytes(b)

    return task_id


def check_model(model_name: str) -> bool:
    # global S2H_TIME
    socket = _get_socket(server_addr=server_addr_map["check_model"])
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    payload_size = int.from_bytes(shm.buf[:HEADER_SIZE], byteorder="little")
    b = shm.buf[int(HEADER_SIZE): int(HEADER_SIZE) + payload_size].tobytes()
    d = json.loads(b)

    task_id = d.get("model_name", "")
    req = mq_types.CheckModelRequest(model_name=model_name, task_id=task_id)
    resp_dict = _send_recv(socket, req)
    return resp_dict["done"]


def reset_status():
    socket = _get_socket(server_addr=server_addr_map["reset_status"])
    rank = _rank_info[os.getpid()]
    req = mq_types.ResetStatusRequest(rank=rank)
    print("Send reset")
    _send_recv(socket, req)


def load_meta(file_name):
    socket = _get_socket(server_addr=server_addr_map["load_meta"])
    req = mq_types.GetMetaRequest(file_name=file_name)
    resp_dict = _send_recv(socket, req)
    return resp_dict["meta_str"]


def _load_tensor_meta(file_name):
    socket = _get_socket(server_addr=server_addr_map["load_tensor_meta"])
    req = mq_types.GetMetaTensorRequest(file_name=file_name)
    resp_dict = _send_recv(socket, req)
    return resp_dict["meta_tensors"]


def print_profile():
    pass
    # stats = pstats.Stats(profiler).sort_stats("cumtime")
    # stats.print_stats()


def recursive_walk_through_directory(directory: str):
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".dangertensors"):
                file_path = os.path.join(root, file)
                meta_path = file_path.replace(".dangertensors", ".meta")
                if os.path.exists(meta_path):
                    file_names.append(file_path)

        for dir in dirs:
            dir_path = os.path.join(root, dir)
            # recursively walk through subdirectories
            sub_file_names = recursive_walk_through_directory(dir_path)
            file_names.extend(sub_file_names)
    return file_names


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
    os.makedirs(f"/tmp/{out_dir}", exist_ok=True)

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


def comfyui_hook(func):
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # ckpt, safe_load=False, device=None, return_metadata=False
        param_names = list(sig.parameters.keys())
        ckpt_file = bound.arguments.get('ckpt')
        ckpt_file = ckpt_file.replace(".safetensors", ".dangertensors")
        device = bound.arguments.get('device', None)
        return_meta = bound.arguments.get('return_metadata', False)
        check_model(ckpt_file)

        print(f"Loading dangertensor from {ckpt_file} to device {device}")

        sd = {}

        tensor_name_vec = _load_tensor_meta(file_name=ckpt_file)
        print(tensor_name_vec[0])
        for item in tensor_name_vec:  # datalength = nele * ele_size
            # create tensor
            # print(item)
            # raise(NotImplementedError)
            name = item['name']
            shape = item['shape']
            shape = ast.literal_eval(shape.replace("torch.Size", ""))
            dtype_str = item['dtype']
            print(f"{name}: {shape}, {dtype_str}, device={device}")
            # if device is None:
            #     device = torch.device("cpu")
            dtype = getattr(torch, dtype_str.split('.')[1])
            tensor = torch.empty(size=torch.Size(shape), dtype=dtype).cuda()
            load_tensor(tensor, name)
            sd[name] = tensor

        if return_meta:
            safetensor_file_meta = load_meta(file_name=ckpt_file)
        else:
            safetensor_file_meta = None

        return (sd, safetensor_file_meta) if return_meta else sd

    return wrapper
