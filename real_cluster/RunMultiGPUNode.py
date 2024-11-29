#!/usr/bin/env python
## -*- coding: utf-8 -*-
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import datasets,transforms,models
import torch.optim as optim
import copy
import argparse
import threading
import time
import datetime
import ast

def ddp_setup(local_rank, args):
    """
       Set up the distributed environment.
    """
    node_rank=args.node_rank
    nprocs_per_node=args.nprocs_per_node
    global_rank = node_rank * nprocs_per_node + local_rank
    # Initialize the process group.
    # 'backend' specifies the communication backend to be used, "nccl" is optimized for GPU training.
    print("world size:",nprocs_per_node * args.nnodes)
    init_process_group(backend="nccl", rank=global_rank, world_size=nprocs_per_node* args.nnodes)

    # Set the current CUDA device to the specified device (identified by rank).
    # This ensures that each process uses a different GPU in a multi-GPU setup.
    if len(args.gpu_id_list)!=0:
        torch.cuda.set_device(args.gpu_id_list[local_rank])
    else:
        torch.cuda.set_device(local_rank)
    

def single_training(local_rank,args,model):
    """
       Main training function for distributed data parallel (DDP) setup.
    """
    # Set up the distributed environment, including setting the master address, port, and backend.
    ddp_setup(local_rank, args)
    #set device for model and data
    model.set_local_rank(local_rank)
    # Load the necessary training objects - dataset, model, and optimizer.
    model.load_mode_data()
    # Train the model for the specified number of epochs.
    model.run()
    # Cleanup the distributed environment after training is complete.
    destroy_process_group()


def parse_list_arg(list_arg):
    try:
        return ast.literal_eval(list_arg)
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid list argument: {list_arg}")

from Recorder import Record
from Model_ResNet_etal import ResNet_etal_class
from Model_Bert import Bert_class
from Model_GCN import GCN_class
from Model_GraphSage import GraphSage_class



def Record_resource(gpu_id, sample_interval,out_dir, out_file_name,event):
    record=Record(gpu_id=gpu_id,sample_interval=sample_interval,out_dir=out_dir, out_file_name=out_file_name,event=event)
    record.run()

def Run_model_training(args_t,dataset_dir):
    if args_t.model_name == "ResNet18" or args_t.model_name == "ResNet50" or args_t.model_name =="AlexNet"\
        or args_t.model_name =="VGG16" or args_t.model_name =="MobileNetv2":
        model=ResNet_etal_class(args_t,dataset_dir)
    elif args_t.model_name == "Bert":
        model=Bert_class(args_t,dataset_dir)
    elif args_t.model_name == "GCN":
        model=GCN_class(args_t,dataset_dir)
    elif args_t.model_name == "GraphSage":
        model=GraphSage_class(args_t,dataset_dir)
    else:
        print("model_name wrong!")
        exit(-1)
    mp.spawn(single_training, args=(args_t,model), nprocs=args_t.nprocs_per_node)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='simple distributed training job')
    #系统参数
    parser.add_argument('--nnodes', default=1, type=int, help='The number of nodes in multi-node training')
    parser.add_argument('--node_rank', default=0, type=int, help='The rank of the node in multi-node training')
    parser.add_argument('--nprocs_per_node', default=1, type=int,help='used gpu number for each node')
    parser.add_argument('--gpu_id_list', default=[], type=parse_list_arg,help='gpu id for each node used')
    #模型通用参数
    parser.add_argument('--model_name',default="ResNet18",help='model name, such as ResNet18, GCN, Bert...')
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--total_epochs', default= 10,type=int, help='Total epochs to train the model')
    parser.add_argument('--worker_num', default= 4,type=int, help='Number of worker for data load')
    #模型特定参数
    parser.add_argument('--squad_data_size',default=0,type=int,help='Size of squad dataset for Bert')
    parser.add_argument('--layer_num',default=10,type=int,help='Layer number for GCN')
    parser.add_argument('--layer_feature',default=10,type=int,help='Layer feature number for GCN')

    #记录参数
    parser.add_argument("--sample_interval", default=1, type=float,help='sample interval for recorder')
    parser.add_argument("--record_flage",default=False, type=bool, help='A flage for if to use recorder to save resource information')
    args = parser.parse_args()

    version="v3"
    print("code version:"+version)

    # 数据集路径
    # 获取当前文件所在目录的上级目录
    parent_dir  = os.path.dirname(os.path.abspath(os.curdir))
    dataset_dir = parent_dir + '/dataset/'
    out_dir     = parent_dir + "/output/"

    # 获取当前时间
    now_time    = datetime.datetime.now()
    # 格式化输出
    formatted_time = now_time.strftime('%m_%d_%H_%M_%S')
    device_name =''
    if torch.cuda.is_available():
        device_name=torch.cuda.get_device_name(0).replace(" ","_")
    else:
        device_name="CPU"

    nnodes=args.nnodes
    node_rank=args.node_rank
    nprocs_per_node=args.nprocs_per_node
    gpu_id_list=args.gpu_id_list

    model_name=args.model_name
    batch_size=args.batch_size
    total_epochs=args.total_epochs
    sample_interval=args.sample_interval
    

    out_file_name=model_name+"-"+device_name+\
        "-nno:"+args.nnodes.__str__()+"-nra:"+args.node_rank.__str__()+"-ppn:"+args.nprocs_per_node.__str__()+\
        "-bs:"+batch_size.__str__() +"-ep:"+total_epochs.__str__() +\
        "-si:"+sample_interval.__str__()+"-tim:"+formatted_time+".txt"
    print(out_file_name)
    if args.record_flage: 
        event=threading.Event()
        subTread_record=threading.Thread(target=Record_resource,args=(-1,sample_interval,out_dir,out_file_name,event))
        subTread_record.start()
    time.sleep(1)
    #主线程
    Run_model_training(args,dataset_dir)
    
    if args.record_flage: 
        event.set()
        subTread_record.join()
    print("process end!")
