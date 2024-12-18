import torch
import numpy as np


print("torch version:", torch.__version__)
print("nccl_version:",torch.cuda.nccl.version())
a=[[2,3],[53,58,9,6],[6,8,9,5]]
for [index, value] in a:
    
    print(f"index:{index}\t value:{value}")