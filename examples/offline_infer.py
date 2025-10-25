from vllm import LLM, SamplingParams

model_path = "your-local-model-path"
tp_size = your-tp-size
prompts = ["haha how are you"]
sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=10)

llm = LLM(model=model_path, tensor_parallel_size = tp_size, enforce_eager=True, max_model_len=4096)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    tokens = output.outputs[0].token_ids
    print(
        f"Prompt: {prompt!r}, Generated text: {generated_text!r}, output tokens: {tokens}"
    )