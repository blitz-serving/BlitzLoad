import torch
import sys
import importlib.util
import inspect
# import blitz_lib
import time

spec = importlib.util.spec_from_file_location(
    "blitz_lib", "build/blitz_lib.cpython-310-x86_64-linux-gnu.so"
)
blitz_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blitz_lib)
print(blitz_lib.__file__)
# # 打印模块中所有类
# for name, obj in inspect.getmembers(blitz_lib, inspect.isclass):
#     print("Class:", name)

# # 打印模块中所有函数
# for name, obj in inspect.getmembers(blitz_lib, inspect.isfunction):
#     print("Function:", name)

# # 打印某个类的方法
# for name, obj in inspect.getmembers(blitz_lib.DataMover, inspect.isfunction):
#     print("DataMover method:", name)


# dm = blitz_lib.datamover()
start = time.time()
blitz_lib.DataMover.init(blitz_lib.Mode.CPU)
blitz_lib.DataMover.register_buffer(5 * 1024 * 1024 * 1024, "blitz")
print(f"Time1: {time.time()-start}")

start = time.time()
blitz_lib.DataMover.load_file_to_buffer_sync(
    "blitz", ["/nvme/models/Meta-Llama-3-8B-Instruct/model-00001-of-00004.safetensors"]
)
print(f"Time2: {time.time()-start}")


t = torch.zeros((4096,), dtype=torch.bfloat16, device="cuda:0")
start = time.time()
blitz_lib.DataMover.load_buffer_to_gpu_sync("blitz", t)
print(f"Time3: {time.time()-start}")

blitz_lib.DataMover.print_shm_info("blitz")
