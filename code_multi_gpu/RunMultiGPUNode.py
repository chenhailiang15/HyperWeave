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


from Recorder import Record
from Model_ResNet_etal import ResNet_etal_class
from Model_Bert import Bert_class
from Model_GCN import GCN_class
from Model_GraphSage import GraphSage_class

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





def Record_resource(args, gpu_id, out_dir, out_file_name,event):
    record=Record(gpu_id=gpu_id,net_card=args.net_card, sample_interval=args.sample_interval,out_dir=out_dir, out_file_name=out_file_name,event=event,print_flage=args.print_flage)
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
    os.environ["NCCL_SOCKET_IFNAME"]="eno1"
    # os.environ["NCCL_DEBUG"]="INFO"

    parser = argparse.ArgumentParser(description='simple distributed training job')
    #系统参数
    parser.add_argument('--nnodes', default=1, type=int, help='The number of nodes in multi-node training')
    parser.add_argument('--node_rank', default=0, type=int, help='The rank of the node in multi-node training')
    parser.add_argument('--nprocs_per_node', default=1, type=int,help='used gpu number for each node')
    parser.add_argument('--gpu_id_list', default=[], type=parse_list_arg,help='gpu id for each node used')
    #模型通用参数
    parser.add_argument('--model_name',default="GraphSage",help='model name, such as ResNet18, GCN, Bert...')
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--total_epochs', default= 10,type=int, help='Total epochs to train the model')
    parser.add_argument('--worker_num', default= 4,type=int, help='Number of worker for data load')
    #模型特定参数
    parser.add_argument('--squad_data_size',default=1000,type=int,help='Size of squad dataset for Bert')
    parser.add_argument('--layer_num',default=10,type=int,help='Layer number for GCN')
    parser.add_argument('--layer_feature',default=10,type=int,help='Layer feature number for GCN')

    #记录参数
    parser.add_argument("--net_card",default="eno1")
    parser.add_argument("--sample_interval", default=1, type=float,help='sample interval for recorder')
    parser.add_argument("--record_flage", action='store_true', help='A flage for if to use recorder to save resource information')
    parser.add_argument("--print_flage", action='store_true')
    #其他参数
    parser.add_argument("--environ_flage", action='store_true')
    args = parser.parse_args()
    #设置
    if not args.environ_flage:
        os.environ["MASTER_ADDR"]="localhost"
        os.environ["MASTER_PORT"]="12355"


    version="v4"
    print("code version:"+version)

    # 数据集路径
    # 获取当前文件所在目录的上级目录
    path=os.path.abspath(os.curdir)
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
    layer_num=args.layer_num
    layer_feature=args.layer_feature
    

    
    print("record flage:",args.record_flage)
    if args.record_flage:
        out_file_name=model_name+"-"+device_name+\
        "-nno:"+args.nnodes.__str__()+"-nra:"+args.node_rank.__str__()+"-ppn:"+args.nprocs_per_node.__str__()+\
        "-bs:"+batch_size.__str__() +"-ep:"+total_epochs.__str__() +"-lan:"+layer_num.__str__() +"-laf:"+layer_feature.__str__() +\
        "-si:"+sample_interval.__str__()+"-tim:"+formatted_time+".txt"
        print(out_file_name)
        event=threading.Event()
        subTread_record=threading.Thread(target=Record_resource,args=(args, -1, out_dir,out_file_name,event))
        subTread_record.start()
    time.sleep(1)
    #主线程
    subTread_record2=threading.Thread(target=Run_model_training,args=(args,dataset_dir))
    # Run_model_training(args,dataset_dir)
    subTread_record2.start()
    subTread_record2.join()
    if args.record_flage: 
        event.set()
        subTread_record.join()
    print("process end!")
