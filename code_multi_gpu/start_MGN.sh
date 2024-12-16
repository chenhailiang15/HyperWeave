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

node_rank=0
net_card="eno1"
MASTER_ADDR="10.26.128.51"   # one node:localhost  multi node: master ip
MASTER_PORT="12355"

model_name="AlexNet"
world_size=3
nprocs_list=[3]

gpu_id_list="[[0,1,2],[]]"


total_epochs=10
batch_size=16           #8 for Bert (default:16)
worker_num=4

squad_data_size=1000
layer_num=10        #5000 for GCN (default:10)
layer_feature=10        #100 for GCN (default:10)

sample_interval=0.1


python WeaveExecutor.py --MASTER_ADDR ${MASTER_ADDR} --MASTER_PORT ${MASTER_PORT} --net_card ${net_card}  --model_name ${model_name} --node_rank ${node_rank} \
--world_size ${world_size} --nprocs_list ${nprocs_list} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
--batch_size ${batch_size} --total_epochs ${total_epochs} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
--sample_interval ${sample_interval} #--record_flage --print_flage

# model_name="ResNet50"
# # batch_size=8
# # layer_num=10
# # layer_feature=10
# python RunMultiGPUNode.py --model_name ${model_name} --node_rank ${node_rank} --nnodes ${nnodes} \
# --nprocs_per_node ${nprocs_per_node} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
# --batch_size ${batch_size} --total_epochs ${total_epochs} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
# --sample_interval ${sample_interval} --environ_flage --record_flage

# model_name="MobileNetv2"
# # batch_size=16
# python RunMultiGPUNode.py --model_name ${model_name} --node_rank ${node_rank} --nnodes ${nnodes} \
# --nprocs_per_node ${nprocs_per_node} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
# --batch_size ${batch_size} --total_epochs ${total_epochs} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
# --sample_interval ${sample_interval} --environ_flage --record_flage

# model_name="GCN"
# layer_num=5000       
# layer_feature=100
# python RunMultiGPUNode.py --model_name ${model_name} --node_rank ${node_rank} --nnodes ${nnodes} \
# --nprocs_per_node ${nprocs_per_node} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
# --batch_size ${batch_size} --total_epochs ${total_epochs} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
# --sample_interval ${sample_interval} --environ_flage --record_flage

# model_name="Bert"
# layer_num=10       
# layer_feature=10
# batch_size=8
# python RunMultiGPUNode.py --model_name ${model_name} --node_rank ${node_rank} --nnodes ${nnodes} \
# --nprocs_per_node ${nprocs_per_node} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
# --batch_size ${batch_size} --total_epochs ${total_epochs} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
# --sample_interval ${sample_interval} --environ_flage --record_flage

# model_name="GraphSage"
# batch_size=16
# python RunMultiGPUNode.py --model_name ${model_name} --node_rank ${node_rank} --nnodes ${nnodes} \
# --nprocs_per_node ${nprocs_per_node} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
# --batch_size ${batch_size} --total_epochs ${total_epochs} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
# --sample_interval ${sample_interval} --environ_flage --record_flage

# model_name="AlexNet" 
# python RunMultiGPUNode.py --model_name ${model_name} --node_rank ${node_rank} --nnodes ${nnodes} \
# --nprocs_per_node ${nprocs_per_node} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
# --batch_size ${batch_size} --total_epochs ${total_epochs} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
# --sample_interval ${sample_interval} --environ_flage --record_flage

# model_name="VGG16"
# python RunMultiGPUNode.py --model_name ${model_name} --node_rank ${node_rank} --nnodes ${nnodes} \
# --nprocs_per_node ${nprocs_per_node} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
# --batch_size ${batch_size} --total_epochs ${total_epochs} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
# --sample_interval ${sample_interval} --environ_flage --record_flage