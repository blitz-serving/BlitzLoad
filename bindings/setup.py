from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")

setup(
    name="blitz_lib",
    ext_modules=[
        CUDAExtension(
            name="blitz_lib",
            sources=[
                "bindings/py_binding.cpp",
                # "include/data_mover.hpp",
                # "include/memory_manager.hpp",
                # "include/logger.h"
            ],  # 你自己的实现文件
            include_dirs=["/nvme/ly/blitz-scale-vllm/include", cuda_home],
            # library_dirs=[os.path.join(cuda_home, "lib64")],
            # libraries=["cudart"],
            extra_compile_args=["-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
