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

**envs**

```bash
echo 'export CMAKE_PREFIX_PATH=/root/.local' >> /root/.bashrc
echo 'export PATH=/root/.local/bin:$PATH' >> /root/.bashrc
source /root/.bashrc

apt update && \
apt install git vim wget curl autoconf pkg-config libssl-dev libzmq3-dev -y && \
apt reinstall libibverbs-dev
```

**lib-blitz**

```bash
git clone --recursive git@github.com:datacanvas-blitzllm/lib-blitz-scale.git
cmake -Bbuild -DTORCH_CUDA_ARCH_LIST="9.0"
cmake --build ./build -j
```

**danger_tensor**: for each model, we should convert safetensor files into dangertensor files

```bash
# you should modify some arguments in make_dangertensors file
# editable args: model_name, model_directory, output_path, tp_size
python lib-blitz-scale/utils/make_dangertensor.py
```

**py-blitz-lib**

blitz_lib python package should be installed in the same env with vllm

```bash
cd blitz_lib
pip install -e .
```

**RUN BLITZ_ENGINE**
```bash
# in directory lib-blitz-scale
./build/mq_server
```

### vLLM

- Install editable vLLM, refer to [vllm_doc](https://docs.vllm.ai/en/v0.9.2/getting_started/installation/gpu.html#build-wheel-from-source), you can checkout from commit ab9f2cfd1942f7ddfee658ce86ea96b4789862af


**Modifies in vLLM code**

from commit ab9f2cfd1942f7ddfee658ce86ea96b4789862af apply changes.patch

```bash
# in vllm directory
git checkout -b blitz ab9f2cfd1942f7ddfee658ce86ea96b4789862af
git apply path-to-changes.patch
```

**RUN ONLINE SERVING**

```bash
# start blitz_engine before run inference test, see RUN BLITZ_ENGINE section
vllm serve <path-to-model>
```

**RUN OFFLINE INFER TEST**

```bash
# start blitz_engine before run inference test, see RUN BLITZ_ENGINE section
python offline_infer.py
```

```python
# offline_infer.py
from vllm import LLM, SamplingParams

model_path = "your-local-model-path"
prompts = ["haha how are you"]
sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=10)

llm = LLM(model=model_path, enforce_eager=True, max_model_len=4096)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    tokens = output.outputs[0].token_ids
    print(
        f"Prompt: {prompt!r}, Generated text: {generated_text!r}, output tokens: {tokens}"
    )
```