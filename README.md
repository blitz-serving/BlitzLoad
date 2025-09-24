# blitz-scale-vllm

## Quick Start

### Engine

```bash
docker run -it \
    --privileged \
    --gpus all \
    --ipc host \
    --network host \
    --workdir /root \
    --name lib-blitz \
    nvcr.io/nvidia/pytorch:24.06-py3 /bin/bash
```

envs

```bash
echo 'export CMAKE_PREFIX_PATH=/root/.local' >> /root/.bashrc
echo 'export PATH=/root/.local/bin:$PATH' >> /root/.bashrc
source /root/.bashrc
```

grpc

```bash
git clone --recursive -b v1.75.0 https://github.com/grpc/grpc.git
cd grpc
mkdir -p cmake/build && \
cd cmake/build
cmake \
    -DgRPC_BUILD_TESTS=OFF \
    -DgRPC_INSTALL=ON \
    -DCMAKE_INSTALL_PREFIX=/root/.local \
    ../..
make -j64
make install
```

lib-blitz

```bash
git clone --recursive git@github.com:blitz-serving/lib-blitz-scale.git
cmake -Bbuild -DTORCH_CUDA_ARCH_LIST="8.0"
cmake --build ./build -j
```

danger_tensor: for each model, we should convert safetensor files into dangertensor files

```bash
# you should modify some necessary params in make_dangertensors file
python lib-blitz-scale/utils/make_dangertensor.py
```

py-blitz-lib

```bash
cd blitz_lib
pip install -e .
```

### vLLM

- Install editable vLLM, refer to [vllm_doc](https://docs.vllm.ai/en/v0.9.2/getting_started/installation/gpu.html#build-wheel-from-source)


modifies in vLLM code

```python
# in llm.py, class LLM __init__
import blitz_lib
blitz_lib.pull_model(model)


# in linear.py
from blitz_lib import vllm_hook
# add @vllm_hook before weight_loader
@vllm_hook
def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
    # some codes


# in vocab_parallel_embedding.py VocabParallelEmbedding' weight_loader
import blitz_lib
blitz_lib.load_weight_from_ipc_handle(param, "")


# in default_loader.py, add func _prepare_dangertensors
def _prepare_dangertensors(self, directory):
    import re
    pattern = re.compile(r"dangertensors\.(\d+)\.meta$")
    files_with_number = []

    for root, _, files in os.walk(directory):
        for f in files:
            match = pattern.match(f)
            if match:
                number = int(match.group(1))
                files_with_number.append((number, os.path.join(root, f)))

    files_with_number.sort(key=lambda x: x[0])

    return [path for _, path in files_with_number]
# in _get_weights_iterator
hf_folder, hf_weights_files, use_safetensors, use_dangetensors = ("", [], False, True)
danger_metas = self._prepare_dangertensors(source.model_or_path)
# some codes...
elif use_dangetensors:
    import blitz_lib
    pull_done = False
    while not pull_done:
        pull_done = blitz_lib.check_model()
    weights_iterator = dangertensors_weights_iterator(danger_metas, self.load_config.use_tqdm_on_load)
```