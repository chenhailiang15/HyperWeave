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
from util import *





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

    #设置
    os.environ["MASTER_ADDR"]=args.MASTER_ADDR
    os.environ["MASTER_PORT"]=args.MASTER_PORT
    # os.environ["NCCL_SOCKET_IFNAME"]=args.net_card
    
    # Set up the distributed environment, including setting the master address, port, and backend.
    ddp_setup(local_rank, args)
    #set device for model and data
    model.set_local_rank(local_rank)
    
    #判断要不要启动同步器
    if local_rank<args.max_sync_num:
        model.set_shm_name(args.prior, args.shm_name_list[local_rank])
    else:
        model.set_shm_name(args.prior, "", enable_flage=False)
    # Load the necessary training objects - dataset, model, and optimizer.
    model.load_mode_data_analyze()
    # Train the model for the specified number of epochs.
    model.run_analyze()
    # Cleanup the distributed environment after training is complete.
    destroy_process_group()

    
def Record_resource(args, gpu_id, out_dir, out_file_name,event):
    record=Record(gpu_id=gpu_id,net_card="", sample_interval=args.sample_interval,out_dir=out_dir, out_file_name=out_file_name,event=event,print_flage=args.print_flage)
    record.run_analyze(args.queue)


def analyze_one_task(args_t,dataset_dir):
    if args_t.model_name == "ResNet18" or args_t.model_name == "ResNet50" or args_t.model_name =="AlexNet"\
        or args_t.model_name =="VGG16" or args_t.model_name =="MobileNetv2":
        model=ResNet_etal_class(args_t,dataset_dir,queue=args.queue)
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
    return


def analyze_tasks(args,dataset_dir):
    args.total_epochs=2
    model_name_list=["AlexNet","ResNet18"]
    batch_size_list=[8,16]
    max_parrallel=2
    for model_name in model_name_list:
        for batch_size in batch_size_list:
            for parrallel in range(1,max_parrallel+1):
                print("start analyze: ", model_name+"-"+batch_size.__str__()+"-"+parrallel.__str__())
                args.queue.append(model_name+"-"+batch_size.__str__()+"-"+parrallel.__str__())
                args.model_name=model_name
                args.batch_size=batch_size
                args.parrallel=parrallel
                analyze_one_task(args,dataset_dir)
                
    return



if __name__=="__main__":
    
    args=args_weave()
    my_queue=[]
    args.set_queue(my_queue)
    dataset_dir=get_dataset_dir()
    output_dir=get_output_dir()
    record_file_name=generate_file_name_for_analyze()
    event=threading.Event()
    subthread_record=threading.Thread(target=Record_resource,args=(args,-1,output_dir,record_file_name,event))
    subthread_record.start()
    analyze_tasks(args,dataset_dir)
    event.set()
    subthread_record.join()
    print("process end!")