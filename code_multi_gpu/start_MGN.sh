#!/bin/bash

export MASTER_ADDR="localhost"
export MASTER_PORT="12355"

model_name="GraphSage"
nnodes=1
node_rank=0
nprocs_per_node=2
gpu_id_list=[1,2,3]
worker_num=4
squad_data_size=1000

python RunMultiGPUNode.py --model_name ${model_name} --node_rank ${node_rank} --nnodes ${nnodes} \
--nprocs_per_node ${nprocs_per_node} --gpu_id_list ${gpu_id_list} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
--environ_flage --record_flage