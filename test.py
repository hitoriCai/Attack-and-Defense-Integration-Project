import torch

a = torch.randn((2,4))
print(a)
_, idx = torch.max(a, dim=1)
print(idx)
print(a.gather(1, idx.unsqueeze(1)))