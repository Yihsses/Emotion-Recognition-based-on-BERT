import torch

# 測試是否有可用的 GPU
cuda_available = torch.cuda.is_available()
print(cuda_available)