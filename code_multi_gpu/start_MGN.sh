#!/bin/bash

#Ber batch size should be about 8

export MASTER_ADDR="10.26.128.70"   # one node:localhost  multi node: master ip
export MASTER_PORT="12355"

model_name="ResNet18"
nnodes=1
node_rank=0
nprocs_per_node=2
gpu_id_list=[]

total_epochs=10
batch_size=16           #8 for Bert (default:16)
worker_num=4

squad_data_size=1000
layer_num=10        #5000 for GCN (default:10)
layer_feature=10        #100 for GCN (default:10)

sample_interval=0.1


python RunMultiGPUNode.py --model_name ${model_name} --node_rank ${node_rank} --nnodes ${nnodes} \
--nprocs_per_node ${nprocs_per_node} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
--batch_size ${batch_size} --total_epochs ${total_epochs} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
--sample_interval ${sample_interval} --environ_flage --record_flage --print_flage

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