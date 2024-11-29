import torch

print("torch version:", torch.__version__)
print("nccl_version:",torch.cuda.nccl.version())