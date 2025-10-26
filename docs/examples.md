# BlitzLoad Examples

This page collects runnable examples that demonstrate how to pair BlitzLoad with common inference stacks. Each example assumes you completed the Quick Start steps in the repository root.

## Offline Inference with vLLM (Single-node startup)

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
   ./build/mq_server
   ```

4. **Run offline inference**
   - Ensure the Blitz engine stays running.
   - Execute the example script from the repository root:
     ```bash
     python examples/offline_infer.py
     ```
   - The script first loads weights through BlitzLoad and then evaluates a batch of prompts with vLLM.

## Online Serving with vLLM (Single-node startup)

1. Start the Blitz engine if it is not already running:
   ```bash
   ./build/mq_server
   ```
2. Launch vLLM with DangerTensor weights:
   ```bash
   vllm serve <path-to-model> \
     --tensor-parallel-size <tp-size> \
     --blitzload-config /path/to/blitz_config.json
   ```
3. Exercise the endpoint using your preferred client (e.g., `curl`, `litellm`, or benchmarking tooling).

For production rollouts, co-locate the Blitz engine with SSD-backed DangerTensor shards and leverage RDMA/NVLink connectivity for remote parameter fetches (to be updated).


## Online Serving with vLLM (w/ RDMA-NVLink optimized transfer)

TBD
