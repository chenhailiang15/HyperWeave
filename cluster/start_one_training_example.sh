#!/bin/bash

system="HyperWeave"
mode="train"

node_rank=0               # Number of nodes participating in training
net_card="eth0"           # Please modified to your Network Card name
MASTER_ADDR="localhost"   # Please modified according to your need (one node:localhost  multi node: master ip)
MASTER_PORT="12365"       # Please modified according to your need (communication port)

model_name="ResNet18"     # model name
world_size=1              # all GPU number for training 
nprocs_list=[1]           # List of the number of GPUs for each node participating in the training.
gpu_id_list="[[0],[]]"    # List of GPU IDs for each node participating in the training.

job_idx=0                 # Job ID               

total_epochs=1           # epoch number
batch_size=256            # batch size 
worker_num=0              # worker number

squad_data_size=1000      # parameter for specifice model
layer_num=10              # 100 for GCN (default:10)
layer_feature=10          # 100 for GCN (default:10)

sample_interval=0.1       # Resource utilization sampling interval.


python platform_h/HyperWeaveExecutor.py --MASTER_ADDR ${MASTER_ADDR} --MASTER_PORT ${MASTER_PORT} --net_card ${net_card}  --model_name ${model_name} --node_rank ${node_rank} \
--world_size ${world_size} --nprocs_list ${nprocs_list} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
--batch_size ${batch_size} --total_epochs ${total_epochs} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
--sample_interval ${sample_interval}  --job_idx ${job_idx} --system ${system} --mode ${mode} --record_flage #--print_flage
