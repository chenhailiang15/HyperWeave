#!/usr/bin/env python
## -*- coding: utf-8 -*-
from Recorder import Record
from Model_ResNet_etal import ResNet_etal_class
import argparse
import os
import time
import torch
import datetime
import threading
import subprocess







def Record_resource(args, gpu_id, out_dir, out_file_name,event):
    record=Record(gpu_id=gpu_id,net_card=args.net_card, sample_interval=args.sample_interval,out_dir=out_dir, out_file_name=out_file_name,event=event,print_flage=args.print_flage)
    record.run()
    
    
def single_training(args,model,barrier):
    
    model.set_local_rank(args.gpu_id)
    # Load the necessary training objects - dataset, model, and optimizer.
    model.load_mode_data_simplify()
    
    barrier.wait()
    
    # Train the model for the specified number of epochs.
    model.run_simplify()
    
    
    
def Run_model_training(args_t,dataset_dir,barrier):
    if args_t.model_name == "ResNet18" or args_t.model_name == "ResNet50" or args_t.model_name =="AlexNet"\
        or args_t.model_name =="VGG16" or args_t.model_name =="MobileNetv2":
        model=ResNet_etal_class(args_t,dataset_dir)
    # elif args_t.model_name == "Bert":
    #     model=Bert_class(args_t,dataset_dir)
    # elif args_t.model_name == "GCN":
    #     model=GCN_class(args_t,dataset_dir)
    # elif args_t.model_name == "GraphSage":
    #     model=GraphSage_class(args_t,dataset_dir)
    else:
        print("model_name wrong!")
        exit(-1)
        
    single_training(args_t,model,barrier)


def set_mps_mode(gpu_index, mps_mode):
    mps_state = 'ON' if mps_mode else 'OFF'
    command = [
        'nvidia-smi', '--mps-control', mps_state, '--gpu-select', str(gpu_index)
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("wrong")
        # print(f'Error setting MPS mode for GPU') #{gpu_index}: {result.stderr}
    else:
        print("right")
        # print(f'MPS mode set to {mps_state} for GPU {gpu_index}')
    return

import ast
def parse_list_arg(list_arg):
    try:
        return ast.literal_eval(list_arg)
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid list argument: {list_arg}")
    
if __name__=="__main__":
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"    
    parser = argparse.ArgumentParser(description='multi model training job')
    #系统参数
    parser.add_argument('--nnodes', default=1, type=int, help='The number of nodes in multi-node training')
    parser.add_argument('--node_rank', default=0, type=int, help='The rank of the node in multi-node training')
    parser.add_argument('--nprocs_per_node', default=1, type=int,help='used gpu number for each node')
    parser.add_argument('--gpu_id_list', default=[], type=parse_list_arg,help='gpu id for each node used')
    parser.add_argument('--gpu_id', default="1", type=str,help='gpu id for each node used')
    parser.add_argument('--thead_num',default=1, type=int,help='model number')
    #模型通用参数
    parser.add_argument('--model_name',default="ResNet18",help='model name, such as ResNet18, GCN, Bert...')
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--total_epochs', default= 1,type=int, help='Total epochs to train the model')
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
    args = parser.parse_args()
    #设置
    
    version="SG v4"
    print("code version:"+version)
    
    
    # #设置MPS模式
    # set_mps_mode(args.gpu_id,True)
    # exit()

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
    barrier = threading.Barrier(args.thead_num+1)
    threads = []

    for i in range(args.thead_num):
        t = threading.Thread(target=Run_model_training, args=(args,dataset_dir,barrier))
        threads.append(t)
        t.start()
    
    barrier.wait()
    start_time=time.time()

    for t in threads:
        t.join()
        
    end_time=time.time()
    
    if args.record_flage: 
        event.set()
        subTread_record.join()
        
    print("model num:"+args.thead_num.__str__()+"\ttraining time (sec):"+ round(end_time-start_time,2).__str__())
    print("process end!")
