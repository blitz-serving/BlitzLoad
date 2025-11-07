# BlitzLoad ComfyUI Examples

This page collects runnable examples that demonstrate how to use BlitzLoad to accelerate startup of a serving instance on engines like ComfyUI. 


## 1. Accelerate ComfyUI startup in online serving by loading models from host DRAM

1. **Prepare ComfyUI**
   
   - Pull ComfyUI codes and checkout to tag `v0.3.68`.
   - Install ComfyUI from source following the [official instructions](https://github.com/comfyanonymous/ComfyUI?tab=readme-ov-file#manual-install-windows-linux).
   - Apply the BlitzLoad integration patch:
     ```bash
     git apply /path/to/BlitzLoad/patches/comfyui.patch
     ```
    - Copy safetensor files into corresponding directories (e.g. /path/to/ComfyUI/models), you can follow the [examples](https://comfyanonymous.github.io/ComfyUI_examples/)

2. **Generate DangerTensor (a fast transfer format with no danger!) weights**
   - Convert the SafeTensor checkpoint:
     ```bash
     python -m utils.make_dangertensor_diffusion --models-path <models-path>
     # parent path of all model weights, e.g. /path/to/ComfyUI/models
     ```

3. Start the Blitz engine if it is not already running:

   ```bash
   ./build/mq_server --devices <devices(e.g. 0,1,2,3)>
   ```

4. Launch ComfyUI with DangerTensor weights:

   ```bash
   python examples/prepare_comfyui.py <models-dir> # Load model weights from local SSD to host DRAM
   python /path/to/ComfyUI/main.py
   ```