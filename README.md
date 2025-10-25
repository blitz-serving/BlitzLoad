# BlitzLoad

**Accelerating Model Cold Start**

BlitzLoad is a lightweight library designed to **drastically reduce model cold start time** with the primary focus of reducing model weight reading time in large-scale inference systems. It is designed with a minimal API such that it can be seamlessly integrated into existing systems like [vLLM](https://github.com/vllm-project/vllm) to accelerate model loading and initialization, especially in distributed or multi-GPU environments.

The key to fast model loading is leveraging hierarchical loading using a transfer-friendly format (that can be seamlessly converted from existing formats like SafeTensor). The supported parameter sources include host DRAM, RDMA+NVLink-linked DRAM, and SSD. Please refer to our paper for details on how we achieve the best possible loading time in various scenarios.

---

* ⚡ **Fast Cold Start**
  Reduce model loading latency by leveraging a distributed model pool and underlying compute fabrics to efficiently allocate and load models across nodes.

* 🔗 **Seamless Integration with vLLM**
  Works as a drop-in enhancement — no need to modify vLLM’s core loading logic.

* 🔄 **Distributed-Aware Design**
  Supports both single-node multi-GPU and multi-node setups, accelerating model initialization across ranks.


## Demo
<div align="center">
<img src="./docs/blitzonvllm.gif" alt="raw vLLM v.s. blitz on vLLM"  height="400">
</div>

## Quick Start

### Build Load Engine

**prerequisite**

We recommend to use NVIDIA's official containers, e.g., `nvcr.io/nvidia/pytorch:24.06-py3`, and a step-by-step instruction can be found at [use_nv_container](https://github.com/blitz-serving/BlitzLoad/blob/main/docs/use_nv_container.md). 

**lib-blitz**

```bash
git clone --recursive git@github.com:datacanvas-blitzllm/lib-blitz-scale.git
cmake -Bbuild -DTORCH_CUDA_ARCH_LIST="9.0" # for h100/h20
cmake --build ./build -j
```

**danger_tensor**: for each model, we should convert safetensor files into dangertensor files, to enable our engine to load model from local-ssd

```bash
# Add model stacked_param_mapping config to utils/models directory, you can find mapping config from vllm
# e.g. add qwen2.5-vl config: open vllm/model_executor/models/qwen2_5_vl.py and find stacked_param_mapping, add the corresponding file in `utils/models` directory
python -m utils.make_dangertensor --model-path <model-path> --output-path <output-path> --tp-size <tp-size>
```

**py-blitz-lib**

blitz_lib python package should be installed in the same env with vllm

```bash
cd blitz_lib && pip install -e .
```

**RUN BLITZ_ENGINE**
```bash
# in directory lib-blitz-scale
./build/mq_server
```

### Testing with vLLM

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
vllm serve <path-to-model> (--tensor-parallel-size <tp-size>)
```

**RUN OFFLINE INFER TEST**

We provide a testing script at [offline_infer.py](https://github.com/blitz-serving/BlitzLoad/blob/main/examples/offline_infer.py), where vLLM loads model weight via BlitzLoad first and then performs offline inference.

```bash
# start blitz_engine before run inference test, see RUN BLITZ_ENGINE section
python offline_infer.py
```

## Roadmap
- Features
  - [] Scale-up/scale-out hybrid bandwidth aggregation

- Integration to serving ecosystem
  - [] Supporting controller to generate distributed load plan within cluster online   
  - [] Co-design with model switch mechanism, e,g, [kvcached](https://github.com/ovg-project/kvcached)

## Citation

If you like BlitzLoad, please cite our paper:

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