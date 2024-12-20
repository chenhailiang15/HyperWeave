import torch
import numpy as np


print("torch version:", torch.__version__)
print("nccl_version:",torch.cuda.nccl.version())
a=[2,36,8,9,5]
b=[1,35,6,10,3]
result=all(max_value>=need_value for max_value, need_value in zip(a, b))
    
print(a)
print(b)
print(result)