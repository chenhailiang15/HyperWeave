import torch
import numpy as np
from blossom import *
import random


print("torch version:", torch.__version__)
print("nccl_version:",torch.cuda.nccl.version())
print("cudnn version:", torch.backends.cudnn.version())





class job:
    def __init__(self):
        return
    

# 示例用法
if __name__ == "__main__":
    run_jobs_dict={}
    job_num=20
    job_list=[]
    for i in range(job_num):
        job={}
        job["num_gpu"]=3
        job['resource_time']=[random.random(),random.random(),random.random(),random.random()]
        job['job_idx']=i
        job['iteration_time']=100
        job_list.append(job)
    run_jobs_dict[3]=job_list
    
    job_num=10
    job_list=[]
    for i in range(job_num):
        job={}
        job["num_gpu"]=1
        job['resource_time']=[random.random(),random.random(),random.random(),random.random()]
        job['job_idx']=i+100
        job['iteration_time']=10
        job_list.append(job)
    run_jobs_dict[1]=job_list
    
    
    
    
    cluster_gpu=10
    packings=Blossom_Same.run(run_jobs_dict, cluster_gpu)
    for i in packings:
        print(f"key:{i}")
        for _pack in packings[i]:    #obj _pack : _Packing
            print(f"value: (next in pack)", end=": ")
            for job in _pack.best_permutation:
                print(job.job_idx,end="\t")
                
            print("")