import torch
from safetensors.torch import save_file

model = torch.load("stories110M.pt")["model"]
tensors = {}
for k, v in model.items():
    print(k, v.shape)
    tensors[k] = v.clone()
save_file(tensors, "stories110M.safetensors")
