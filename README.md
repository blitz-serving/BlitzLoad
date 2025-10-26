# BlitzLoad:  Accelerate Large Model Cold Start

BlitzLoad is a lightweight model checkpoint loading engine that minimizes the latency of bringing large language models online. It loads model parameters from host DRAM, NVLink/RDMA peers, and local SSDs using a hierarchy-aware highly-optimized multicast plan (described in the BlitzScale system, see https://www.usenix.org/conference/osdi25/presentation/zhang-dingyan). More importantly, it can seamlessly integrate with existing inference stacks such as [vLLM](https://github.com/vllm-project/vllm) and initialize at scale with minimal changes.

## Key Capabilities
- ⚡ **Fast cold starts** through hierarchical weight placement and prefetch pipelines.
- 🔗 **Engine-friendly integration** that keeps vLLM and similar runtimes unmodified.
- 🔄 **Distributed aware** planning for both single-node multi-GPU and multi-node clusters.

## Demo
<div align="center">
<img src="./docs/blitzonvllm.gif" alt="BlitzLoad on top of vLLM" height="400">
</div>

## Quick Start

**1. Prerequisites**

- CUDA-capable GPUs (H100/H20 recommended) and NCCL-compatible fabric if running multi-node.
- NVIDIA PyTorch container such as `nvcr.io/nvidia/pytorch:24.06-py3` or an equivalent environment (see `docs/use_nv_container.md`).

**2. Steps**

1. Clone the repo, initialize submodules, and enter the workspace:
   ```bash
   git clone --recursive https://github.com/blitz-serving/BlitzLoad.git
   cd BlitzLoad
   ```
2. Build the C++ loading engine:
   ```bash
   cmake -B build -DTORCH_CUDA_ARCH_LIST="9.0"
   cmake --build build -j
   ```
3. Install the Python bindings alongside your serving stack:
   ```bash
   pip install -e blitz_lib
   ```
4. Convert model weights to the BlitzLoad DangerTensor format (see `docs/examples.md` for detailed guidance):
   ```bash
   python -m utils.make_dangertensor --model-path <model> --output-path <output> --tp-size <tp>
   ```
5. Launch the Blitz engine prior to service startup:
   ```bash
   ./build/mq_server
   ```
6. Start vLLM (or another compatible runtime) configured to consume BlitzLoad weights. Reference `docs/examples.md` for the required patch and usage pattern.

## Documentation
- `docs/examples.md` — detailed examples for running offline inference and integrating with vLLM.
- `docs/use_nv_container.md` — recommended NVIDIA container workflow.

## Roadmap
- Features (port implementation from https://github.com/blitz-serving/blitz-scale)
  - [ ] Port RDMA/NVLink hybrid bandwidth aggregation.
  - [ ] Add a controller that generates cluster-wide load plans online.
- Serving Ecosystem
  - [ ] Co-design with model switch control planes (e.g., [kvcached](https://github.com/ovg-project/kvcached)) to eliminate the overhead of engine's control plane. 
  - [ ] Integrate with SGLang.
- More application scenarios 
  - [ ] Examples in supporting checkpoint read in post-training. 

## Citation
If you use BlitzLoad in your work, please cite:

```bibTex
@inproceedings{10.5555/3767901.3767917,
author = {Zhang, Dingyan and Wang, Haotian and Liu, Yang and Wei, Xingda and Shan, Yizhou and Chen, Rong and Chen, Haibo},
title = {BLITZSCALE: fast and live large model autoscaling with O(1) host caching},
year = {2025},
isbn = {978-1-939133-47-2},
publisher = {USENIX Association},
address = {USA},
booktitle = {Proceedings of the 19th USENIX Conference on Operating Systems Design and Implementation},
articleno = {16},
numpages = {19},
location = {Boston, MA, USA},
series = {OSDI '25}
}
```
