#!/bin/bash

#Ber batch size should be about 8

# Is_Master=false

# if [ "${Is_Master}" = true ]; then
#     node_rank=0
#     net_card="eno2"
# else
#     node_rank=1
#     net_card="eno1"
# fi
# conda activate torch_mp_chl
system="Weave"
mode="train"

node_rank=0
net_card="eno1"
MASTER_ADDR="localhost"   # one node:localhost  multi node: master ip
MASTER_PORT="12345"

model_name="ResNet18"
world_size=2
nprocs_list=[2]
gpu_id_list="[[0,1],[]]"

job_idx=10

total_epochs=2
batch_size=16           #8 for Bert (default:16)
worker_num=4

squad_data_size=1000
layer_num=10        #5000 for GCN (default:10)
layer_feature=10        #100 for GCN (default:10)

sample_interval=0.1



python WeaveExecutor.py --MASTER_ADDR ${MASTER_ADDR} --MASTER_PORT ${MASTER_PORT} --net_card ${net_card}  --model_name ${model_name} --node_rank ${node_rank} \
--world_size ${world_size} --nprocs_list ${nprocs_list} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
--batch_size ${batch_size} --total_epochs ${total_epochs} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
--sample_interval ${sample_interval}  --job_idx ${job_idx} --system ${system} --mode ${mode} #--record_flage #--print_flage


# model_name="Bert"
# python WeaveExecutor.py --MASTER_ADDR ${MASTER_ADDR} --MASTER_PORT ${MASTER_PORT} --net_card ${net_card}  --model_name ${model_name} --node_rank ${node_rank} \
# --world_size ${world_size} --nprocs_list ${nprocs_list} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
# --batch_size ${batch_size} --total_epochs ${total_epochs} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
# --sample_interval ${sample_interval} 

# model_name="Transformer"
# python WeaveExecutor.py --MASTER_ADDR ${MASTER_ADDR} --MASTER_PORT ${MASTER_PORT} --net_card ${net_card}  --model_name ${model_name} --node_rank ${node_rank} \
# --world_size ${world_size} --nprocs_list ${nprocs_list} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
# --batch_size ${batch_size} --total_epochs ${total_epochs} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
# --sample_interval ${sample_interval} 


# model_name="GCN"
# python WeaveExecutor.py --MASTER_ADDR ${MASTER_ADDR} --MASTER_PORT ${MASTER_PORT} --net_card ${net_card}  --model_name ${model_name} --node_rank ${node_rank} \
# --world_size ${world_size} --nprocs_list ${nprocs_list} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
# --batch_size ${batch_size} --total_epochs ${total_epochs} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
# --sample_interval ${sample_interval} 

# model_name="GraphSage"
# python WeaveExecutor.py --MASTER_ADDR ${MASTER_ADDR} --MASTER_PORT ${MASTER_PORT} --net_card ${net_card}  --model_name ${model_name} --node_rank ${node_rank} \
# --world_size ${world_size} --nprocs_list ${nprocs_list} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
# --batch_size ${batch_size} --total_epochs ${total_epochs} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
# --sample_interval ${sample_interval} 
