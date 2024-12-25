#!/usr/bin/env python
## -*- coding: utf-8 -*-
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import os
import torch
import argparse
import threading
import time
import datetime
import ast
import gc

from Recorder import Record
from util import *
# torch.backends.cudnn.enabled = False
from models.Framework import model_framework

def ddp_setup(local_rank, args):
    """
       Set up the distributed environment.
    """
    node_rank=args.node_rank
    global_rank=0
    for i in range(node_rank):
        global_rank+=args.nprocs_list[i]
    global_rank += local_rank
    # Initialize the process group.
    # 'backend' specifies the communication backend to be used, "nccl" is optimized for GPU training.
    
    init_process_group(backend="nccl", rank=global_rank, world_size=args.world_size)

    # Set the current CUDA device to the specified device (identified by rank).
    # This ensures that each process uses a different GPU in a multi-GPU setup.
    if len(args.gpu_id_list)!=0:
        torch.cuda.set_device(args.gpu_id_list[args.node_rank][local_rank])
    else:
        torch.cuda.set_device(local_rank)
    

def single_training(local_rank,args):
    """
       Main training function for distributed data parallel (DDP) setup.
    """
    # Set up the distributed environment, including setting the master address, port, and backend.
    print("ddp setup...")
    ddp_setup(local_rank, args)
    #set device for model and data
    model_frame=model_framework(local_rank, args)
    
    print("model sync setup...")
    #判断要不要启动同步器
    if args.node_rank in args.shm_name_list:
        gpu_id=args.gpu_id_list[args.node_rank][local_rank]
        if gpu_id in args.shm_name_list[args.node_rank]:
            model_frame.set_shm_name( args.shm_name_list[args.node_rank][gpu_id], args.prior)
        else:
            model_frame.set_shm_name("", args.prior, enable_flage=False)
    else:
        model_frame.set_shm_name("", args.prior, enable_flage=False)
    # Load the necessary training objects - dataset, model, and optimizer.
    model_frame.load_mode_data()
    # Train the model for the specified number of epochs.
    model_frame.run()
    # Cleanup the distributed environment after training is complete.
    destroy_process_group()

def parse_list_shm(list_arg):
    list_arg=list_arg.replace("]","").replace("[","").split(",")
    shm_list=[]
    for shm in list_arg:
        shm_list.append(shm)
        
    return shm_list
    
    
    
    
    
def parse_list_arg(list_arg):
    
    try:
        return ast.literal_eval(list_arg)
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid list argument: {list_arg}")

def Record_resource(args, gpu_id, out_dir, out_file_name,event):
    record=Record(gpu_id=gpu_id,net_card=args.net_card, sample_interval=args.sample_interval,out_dir=out_dir, out_file_name=out_file_name,event=event,print_flage=args.print_flage)
    record.run()

def Run_model_training(args_t):
    
    mp.spawn(single_training, args=(args_t,), nprocs=args_t.nprocs_list[args_t.node_rank])


if __name__=="__main__":
    print("//////////////////////////////////////////////start one job!///////////////////////////////////////////////")
    # os.environ["NCCL_DEBUG"]="INFO"
    
    parser = argparse.ArgumentParser(description='simple distributed training job')
    #优先级参数
    parser.add_argument('--prior', action='store_true',help='A flage for label it is prior to run or not in Synchronizer')
    #系统参数
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--nprocs_list', default=[1,0], type=parse_list_arg)
    parser.add_argument('--node_rank', default=0, type=int, help='The rank of the node in multi-node training')
    parser.add_argument('--gpu_id_list', default=[[0,1],[]], type=parse_list_arg,help='gpu id for each node used')
    # parser.add_argument('--nnodes', default=1, type=int, help='The number of nodes in multi-node training')
    # parser.add_argument('--nprocs_per_node', default=1, type=int,help='used gpu number for each node')
    # parser.add_argument('--gpu_id_list', default=[], type=parse_list_arg,help='gpu id for each node used')
    
    #模型通用参数
    parser.add_argument('--model_name',default="GCN",help='model name, such as ResNet18, GCN, Bert...')
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--batch_num', default=16, type=int)
    parser.add_argument('--total_epochs', default=2,type=int, help='Total epochs to train the model')
    parser.add_argument('--worker_num', default= 4,type=int, help='Number of worker for data load')
    
    #模型特定参数
    parser.add_argument('--squad_data_size',default=1000,type=int,help='Size of squad dataset for Bert')
    parser.add_argument('--layer_num',default=10,type=int,help='Layer number for GCN')
    parser.add_argument('--layer_feature',default=10,type=int,help='Layer feature number for GCN')

    #记录参数
    parser.add_argument("--sample_interval", default=1, type=float,help='sample interval for recorder')
    parser.add_argument("--record_flage", action='store_true', help='A flage for if to use recorder to save resource information')
    parser.add_argument("--print_flage", action='store_true')
    
    #同步参数
    # parser.add_argument("--max_sync_num",default=0,type=int)
    parser.add_argument("--shm_name_list",default="{}",type=parse_list_arg)
    
    #其他参数
    parser.add_argument("--MASTER_ADDR",default="localhost")
    parser.add_argument("--MASTER_PORT",default="12355")
    parser.add_argument("--net_card",default="eno1")
    parser.add_argument("--print_level",default=10, type=int)

    args = parser.parse_args()
    args=args_weave(args)
    
    
    os.environ["MASTER_ADDR"]=args.MASTER_ADDR
    os.environ["MASTER_PORT"]=args.MASTER_PORT
    os.environ["NCCL_SOCKET_IFNAME"]=args.net_card
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:128"
    gc.collect()
    torch.cuda.empty_cache()
    
    if args.print_level>0:
        print("addr:",args.MASTER_ADDR,"port:",args.MASTER_PORT,"netcard:",args.net_card)
    
    # version="v4"
    # print("code version:"+version)

    # 数据集路径
    # 获取当前文件所在目录的上级目录
    path=os.path.abspath(os.curdir)
    parent_dir  = os.path.dirname(os.path.abspath(os.curdir))
    dataset_dir = parent_dir + '/dataset/'
    out_dir     = parent_dir + "/output/"

    
    args.dataset_dir=dataset_dir
    
    # print("record flage:",args.record_flage)
    if args.record_flage:
        
        # 获取当前时间
        now_time    = datetime.datetime.now()
        # 格式化输出
        formatted_time = now_time.strftime('%m_%d_%H_%M_%S')
        device_name =''
        if torch.cuda.is_available():
            device_name=torch.cuda.get_device_name(0).replace(" ","_")
        else:
            device_name="CPU"
        
        
        nnodes=len(args.nprocs_list)
        node_rank=args.node_rank
        
        gpu_id_list=args.gpu_id_list

        model_name=args.model_name
        batch_size=args.batch_size
        total_epochs=args.total_epochs
        sample_interval=args.sample_interval
        layer_num=args.layer_num
        layer_feature=args.layer_feature
        
        
        
        out_file_name=model_name+"-"+device_name+\
        "-nno:"+nnodes.__str__()+"-nra:"+args.node_rank.__str__()+"-wds:"+args.world_size.__str__()+\
        "-bs:"+batch_size.__str__() +"-ep:"+total_epochs.__str__() +"-lan:"+layer_num.__str__() +"-laf:"+layer_feature.__str__() +\
        "-si:"+sample_interval.__str__()+"-tim:"+formatted_time+".csv"
        print(out_file_name)
        event=threading.Event()
        subTread_record=threading.Thread(target=Record_resource,args=(args, 0, out_dir,out_file_name,event))
        subTread_record.start()
        time.sleep(1)
    #主线程
    Run_model_training(args)
    
    if args.record_flage: 
        event.set()
        subTread_record.join()
        
    print("//////////////////////////////////////////////complete one job!///////////////////////////////////////////////")
