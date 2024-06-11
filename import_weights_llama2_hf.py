import torch
from safetensors import safe_open
from safetensors.torch import save_file

layers = 32
all_tensors = {}
for d in (1, 2):
    _filename = f"model-0000{d}-of-00002.safetensors"
    with safe_open(_filename, framework="pt") as fobj:
        for k in fobj.keys():
            all_tensors[k] = fobj.get_tensor(k)

tensors = {
        "tok_embeddings.weight": all_tensors["model.embed_tokens.weight"],
        "output.weight": all_tensors["lm_head.weight"],
        "norm.weight": all_tensors["model.norm.weight"],
}
for k, v in all_tensors.items():
    print(k, v.shape)

for i in range(layers):
    src_prefix = f"model.layers.{i}"
    dst_prefix = f"layers.{i}"
    for c in ("q", "k", "v", "o"):
        tensors[f"{dst_prefix}.attention.w{c}.weight"] = all_tensors[f"{src_prefix}.self_attn.{c}_proj.weight"]
    tensors[f"{dst_prefix}.feed_forward.w2.weight"] = all_tensors[f"{src_prefix}.mlp.down_proj.weight"]
    tensors[f"{dst_prefix}.feed_forward.w1.weight"] = all_tensors[f"{src_prefix}.mlp.gate_proj.weight"]
    tensors[f"{dst_prefix}.feed_forward.w3.weight"] = all_tensors[f"{src_prefix}.mlp.up_proj.weight"]
    tensors[f"{dst_prefix}.attention_norm.weight"] = all_tensors[f"{src_prefix}.input_layernorm.weight"]
    tensors[f"{dst_prefix}.ffn_norm.weight"] = all_tensors[f"{src_prefix}.post_attention_layernorm.weight"]

save_file(tensors, "llama2-7b.safetensors")
