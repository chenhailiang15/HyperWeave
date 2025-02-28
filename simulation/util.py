import datetime
import torch
import os
import ast
import argparse
import string 
import secrets
import psutil
import socket
import json
import numpy as np
import subprocess
import math
# 1.34, 1.29, 1.27, 1.33, 1.3, 1.3, 1.38, 1.37, 1.47, 1.59
mps_time_extend={"AlexNet":[1.03, 1.08, 1.07, 1.08, 1.16, 1.32, 1.47, 1.63, 1.76, 1.94],
                 "ResNet18":[1.01, 1.24, 1.27, 1.22, 1.25, 1.4, 1.53, 1.75, 1.89, 2.15],
                 "ResNet50":[1.02, 1.02, 1.15, 1.4, 1.65, 1.93, 2.16, 2.5, 2.74],
                 "MobileNetv2":[1.0, 1.1, 1.09, 1.14, 1.11, 1.11, 1.18, 1.17, 1.26, 1.36],
                 "VGG16":[1.0, 1.67, 2.33, 3.08, 3.75, 4.52, 5.22, 5.95],
                 "Bert":[1.01, 1.2, 1.42, 1.65, 1.84, 2.08],
                 "Transformer":[1, 1.15, 1.51, 1.9, 2.26, 2.69, 3.03, 3.47, 3.84, 4.29],
                 "GCN":[1.0, 1, 1.06, 1.01, 1.05, 1.13, 1.23, 1.41, 1.42, 1.54],
                 "GraphSage":[1.01, 1.09, 1.71, 2.64, 3.89, 4.9, 5.74, 6.66, 7.44, 8.41]}

gpu_mem_dict={"T4":16.0,
              "MISC":16.0,
              "P100":16.0,
              "V100":16.0,
              "V100M32":32.0,
              "A100M40":40.0,
              "A100M80":80.0
              }

model_to_batch_size_g={"AlexNet" : [8,16,32,64,128],
                         "ResNet18" : [8,16,32,64,128],
                         "ResNet50" : [8,16,32,64,128],
                         "MobileNetv2" : [8,16,32,64,128],
                         "VGG16" : [8,16,32,64,128],
                         "Bert": [8,16],
                         "Transformer": [8,16,32,64],
                         "GCN": [8,16,32,64,128],
                         "GraphSage":[8,16,32,64,128]
                         }

model_list_g=["AlexNet", "ResNet18", "ResNet50", "MobileNetv2", "VGG16", "Bert", "Transformer", "GCN", "GraphSage" ]


def fit_mps_time_delay():
    # 找到最大长度
    max_length = max(len(values) for values in mps_time_extend.values())

    # 初始化平均列表
    average_values = [0] * max_length
    counts = [0] * max_length

    # 计算每个位置的总和与计数
    for values in mps_time_extend.values():
        for i, val in enumerate(values):
            average_values[i] += val
            counts[i] += 1

    # 计算平均值
    for i in range(max_length):
        if counts[i] > 0:
            average_values[i] /= counts[i]

    # 生成对应的 x 索引
    x = np.arange(1,len(average_values)+1)

    # 进行线性拟合
    coefficients = np.polyfit(x, average_values, 1)
    a = coefficients[0]  # 斜率
    b = coefficients[1]  # 截距
    return a,b
    
    


def start_MPS(password):
    command="nvidia-cuda-mps-control -d"
    (status, result)=subprocess.getstatusoutput('echo %s| sudo -S %s' %(password,command))
    print(status)
    print(result)
    
    if (status==1 and "An instance of this daemon is already running" in result) or (status==0 and "nvidia-cuda-mps-control -d" in result and "nvidia-cuda-mps-server" in result):
        
        return True
    else:
        return False
    
    
def stop_MPS(password):
    command="echo quit | nvidia-cuda-mps-control"
    (status, result)=subprocess.getstatusoutput('echo %s| sudo -S %s' %(password,command))
    print(status)
    print(result)
    if (status==1 and "Cannot find MPS control daemon process" in result) or (status==0 and "nvidia-cuda-mps-control -d" not in result and "nvidia-cuda-mps-server" not in result):
        return True
    else:
        return False
    



def analyze_datas(data):
    if len(data) >0:
        np_data=np.array(data)
        _mean=np_data.mean()
        _min=np_data.min()
        _max=np_data.max()
        per_50=np.percentile(np_data, 50)
        per_90=np.percentile(np_data, 90)
        per_95=np.percentile(np_data, 95)
    else:
        _mean=0
        _min=0
        _max=0
        per_50=0
        per_90=0
        per_95=0
    return len(data), round(_mean,2), round(_min,2), round(_max,2), round(per_50,2), round(per_90,2), round(per_95,2)

#判断端口是否被使用
def is_port_in_use(ip, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # 尝试绑定到指定的主机和端口
        result = sock.connect_ex((ip, int(port)))
        return result == 0

#生成共享内存名字
def generate_shm_name(length=10):
    return " "
    # 选择字母和数字
    characters = string.ascii_letters + string.digits
    # 使用secrets.choice从characters中随机选择字符，并使用join将它们组合成一个字符串
    secure_random_string = ''.join(secrets.choice(characters) for i in range(length))
    return secure_random_string

# #解析输入参数为dict
# def parse_dict_shm(dict_arg):

#     shm_dict=ast.literal_eval(dict_arg)
#     return shm_dict
    
#解析输入参数为list 普遍
def parse_list_arg(list_arg):
    try:
        return ast.literal_eval(list_arg)
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid list argument: {list_arg}")

#获取数据集文件夹路径
def get_dataset_dir():
    # 获取当前文件所在目录的上级目录
    cur_dir=os.path.dirname(os.path.abspath(__file__))
    parent_dir  = os.path.dirname(os.path.abspath(cur_dir))
    dataset_dir = parent_dir + '/dataset/'
    return dataset_dir

#获取输出文件夹路径
def get_output_dir():
    cur_dir=os.path.dirname(os.path.abspath(__file__))
    parent_dir  = os.path.dirname(os.path.abspath(cur_dir))
    out_dir     = parent_dir + "/output/"
    return out_dir

#为分析结果生成文件名
def generate_file_name_for_analyze():
    # 获取当前时间
    now_time= datetime.datetime.now()
    # 格式化输出
    formatted_time = now_time.strftime('%m_%d_%H_%M_%S')
    device_name =''
    if torch.cuda.is_available():
        device_name=torch.cuda.get_device_name(0).replace(" ","_")
    else:
        device_name="CPU"
    
    out_file_name="Analyzer-"+device_name+"-tim_"+formatted_time+".csv"
    print(out_file_name)
    return out_file_name

#系统参数解析，便于修改操作
class args_weave:
    def __init__(self,args=None):
        if args==None:
            self.init_with_default()
        else:
            self.init_with_args(args)
        
    def init_with_args(self, args):
         
        
        self.prior=args.prior
        
        
        self.system=args.system
        self.mode=args.mode
        self.job_idx=args.job_idx
        self.idx_on_gpu=args.idx_on_gpu

        #系统参数
        self.world_size=args.world_size
        self.node_rank=args.node_rank
        self.nprocs_list=args.nprocs_list
        self.gpu_id_list=args.gpu_id_list
        
        #模型通用参数
        self.model_name=args.model_name
        self.batch_size=args.batch_size
        self.batch_num=args.batch_num
        self.total_epochs=args.total_epochs
        self.worker_num=args.worker_num
        
        #模型特定参数
        self.squad_data_size=args.squad_data_size
        self.layer_num=args.layer_num
        self.layer_feature=args.layer_feature

        #记录参数
        self.sample_interval=args.sample_interval
        self.record_flage=args.record_flage
        self.print_flage=args.print_flage
        
        #同步参数
        self.max_sync_num=args.max_sync_num
        self.shm_name_list=args.shm_name_list
        
        #其他参数
        self.MASTER_ADDR=args.MASTER_ADDR
        self.MASTER_PORT=args.MASTER_PORT
        self.net_card=args.net_card
        
        self.print_level=args.print_level
        
        
        self.device=None
        self.dataset_dir=None
        

        
    def init_with_default(self):
        self.prior=True
        
        #
        self.system="Weave"
        self.mode="train"
        self.job_idx=0
        self.idx_on_gpu=0
        
        #系统参数
        self.nnodes=1
        self.node_rank=0
        self.nprocs_list=[1]
        self.gpu_id_list=[]
        
        #模型通用参数
        self.model_name="AlexNet"
        self.batch_size=16
        self.batch_num=100        #只有GCN使用
        self.total_epochs=10
        self.worker_num= 4
        
        #模型特定参数
        self.squad_data_size=1000
        self.layer_num=10
        self.layer_feature=10

        #记录参数
        self.sample_interval=1
        self.record_flage=True
        self.print_flage=False
        
        #同步参数
        self.max_sync_num=0
        self.shm_name_list={}
        
        #其他参数
        self.MASTER_ADDR="localhost"
        self.MASTER_PORT="12355"
        self.net_card="eno1"
        
        self.device=None
        self.dataset_dir=None
        self.muri_file_path_name=None
        
        self.overshared_factor=3
        self.strategy="FIFO"
        self.print_level=11
        self.node_kind="4*3090"
        self.sync_flage="True"
        self.mps_flage="True"
        self.validation="True"
        self.model_kind="all_model"
        self.node_num=100
        self.job_num=100
        self.job_together_flage="True"
        self.gpu_mem_percent=0.9
        self.couple_init_iter_percent=0.2
        self.bucket_length=10000
        self.trace_id=0
        self.job_come_time_factor=1
        self.write_trace=False
        self.write_sum=False
        self.gpu_kind="None"
        
    def set_queue(self, queue):
        self.queue=queue
    def set_event(self, event):
        self.event=event
    # def set_shm_name(self, shm_name):
    #     self.shm_name=shm_name
    
    
    
