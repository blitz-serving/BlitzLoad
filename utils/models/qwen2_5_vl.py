stacked_param_mapping = [
    # (param_name, shard_name, shard_id)
    ("attn.qkv.", "attn.q.", "q"),
    ("attn.qkv.", "attn.k.", "k"),
    ("attn.qkv.", "attn.v.", "v"),
    ("mlp.gate_up_proj.", "mlp.gate_proj.", 0),
    ("mlp.gate_up_proj.", "mlp.up_proj.", 1),
]
