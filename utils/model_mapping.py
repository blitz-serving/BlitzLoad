model_stack_mapping = {
    "qwen3": [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ],
    "qwen2": [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ],
    "qwen2.5-vl": [
        # (param_name, shard_name, shard_id)
        ("attn.qkv.", "attn.q.", "q"),
        ("attn.qkv.", "attn.k.", "k"),
        ("attn.qkv.", "attn.v.", "v"),
        ("mlp.gate_up_proj.", "mlp.gate_proj.", 0),
        ("mlp.gate_up_proj.", "mlp.up_proj.", 1),
    ],
    "qwen2-vl": [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    ],
}

def match_model_mapping(model_name: str):
    for key in model_stack_mapping.keys():
        if key in model_name.lower():
            return model_stack_mapping[key]
    return []