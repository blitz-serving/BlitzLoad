# BlitzLoad vLLM Examples

This page collects runnable examples that demonstrate how to use BlitzLoad to accelerate startup of a serving instance on engines like vLLM. 

## 1. Accelerate vLLM startup in online serving by loading models from host DRAM

1. **Prepare vLLM**
   
   - Checkout to commit `ab9f2c` (tested baseline).
   - Install vLLM from source following the [official instructions](https://docs.vllm.ai/en/v0.9.2/getting_started/installation/gpu.html#build-wheel-from-source).
   - Apply the BlitzLoad integration patch:
     ```bash
     git apply /path/to/BlitzLoad/changes.patch
     ```
   
2. **Generate DangerTensor (a fast transfer format with no danger!) weights**
   - Add the `stacked_param_mapping` configuration for your model under `utils/models`. You can copy the mapping from the corresponding `vllm/model_executor/models/<model>.py`.
   - Convert the SafeTensor checkpoint:
     ```bash
     python -m utils.make_dangertensor \
       --model-path <model-path> \
       --output-path <output-path> \
       --tp-size <tensor-parallel-size>
     ```

3. Start the Blitz engine if it is not already running:

   ```bash
   ./build/mq_server --devices <devices(e.g. 0,1,2,3)>
   ```

4. Launch vLLM with DangerTensor weights:

   ```bash
   python examples/prepare_vllm.py <model-path> <tp-size> # Load model weights from local SSD to host DRAM
   vllm serve <path-to-model> \
     --tensor-parallel-size <tp-size>
   ```

5. Exercise the endpoint using your preferred client (e.g., `curl`, `litellm`, or benchmarking tooling).



---



## 2. Accelerate vLLM startup in online serving by loading models from remote memory via RDMA/NVLink

TBD



---


## 3. Offline Inference with vLLM (Single-node startup)

1. **Prepare vLLM**
   
   - Install vLLM from source following the [official instructions](https://docs.vllm.ai/en/v0.9.2/getting_started/installation/gpu.html#build-wheel-from-source).
   - Checkout commit `ab9f2cfd1942f7ddfee658ce86ea96b4789862af` (tested baseline).
   - Apply the BlitzLoad integration patch:
     ```bash
     git checkout -b blitz ab9f2cfd1942f7ddfee658ce86ea96b4789862af
     git apply /path/to/BlitzLoad/changes.patch
     ```
   
2. **Generate DangerTensor (a fast transfer format with no danger!) weights**
   - Add the `stacked_param_mapping` configuration for your model under `utils/models`. You can copy the mapping from the corresponding `vllm/model_executor/models/<model>.py`.
   - Convert the SafeTensor checkpoint:
     ```bash
     python -m utils.make_dangertensor \
       --model-path <model-path> \
       --output-path <output-path> \
       --tp-size <tensor-parallel-size>
     ```

3. **Start the Blitz engine (If not started)**
   ```bash
   ./build/mq_server --devices <devices(e.g. 0,1,2,3)>
   ```

4. **Run offline inference**
   - Ensure the Blitz engine stays running.
   - Execute the example script from the repository root:
     ```bash
     python examples/prepare_vllm.py <model-path> <tp-size> # Load model weights from local SSD to DRAM
     python examples/offline_infer.py # Transfer the parameters from DRAM to the VRAM registered by the engine, and finally load them into vLLM.
     ```
   - The script first loads weights through BlitzLoad and then evaluates a batch of prompts with vLLM.

