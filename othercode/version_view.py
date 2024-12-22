import torch
import numpy as np


print("torch version:", torch.__version__)
print("nccl_version:",torch.cuda.nccl.version())
print("cudnn version:", torch.backends.cudnn.version())
