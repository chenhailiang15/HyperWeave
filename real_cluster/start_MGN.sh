#!/bin/bash

export MASTER_ADDR="localhost"
export MASTER_PORT="12355"

model_name="GraphSage"
nnodes=1
node_rank=0
nprocs_per_node=2
record_flage=True
gpu_id_list=[]

python RunMultiGPUNode.py --model_name ${model_name} --node_rank ${node_rank} --nnodes ${nnodes} --nprocs_per_node ${nprocs_per_node} \
--gpu_id_list ${gpu_id_list} --record_flage ${record_flage} 