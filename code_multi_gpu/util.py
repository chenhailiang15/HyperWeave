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
    parent_dir  = os.path.dirname(os.path.abspath(os.curdir))
    dataset_dir = parent_dir + '/dataset/'
    return dataset_dir

#获取输出文件夹路径
def get_output_dir():
    parent_dir  = os.path.dirname(os.path.abspath(os.curdir))
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
        #系统参数
        self.world_size=args.world_size
        self.node_rank=args.node_rank
        self.nprocs_list=args.nprocs_list
        self.gpu_id_list=args.gpu_id_list
        
        #模型通用参数
        self.model_name=args.model_name
        self.batch_size=args.batch_size
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
        # self.max_sync_num=args.max_sync_num
        self.shm_name_list=args.shm_name_list
        
        #其他参数
        self.MASTER_ADDR=args.MASTER_ADDR
        self.MASTER_PORT=args.MASTER_PORT
        self.net_card=args.net_card
        self.print_level=args.print_level
        
        
        self.device=None
        self.dataset_dir=None
        self.idx=0
        self.mode="train"

        
    def init_with_default(self):
        self.prior=True
        #系统参数
        self.nnodes=1
        self.node_rank=0
        self.nprocs_list=[1]
        self.gpu_id_list=[]
        
        #模型通用参数
        self.model_name="AlexNet"
        self.batch_size=16
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
        self.shm_name_list=[]
        
        #其他参数
        self.MASTER_ADDR="localhost"
        self.MASTER_PORT="12355"
        self.net_card="eno1"
        
        
    def set_queue(self, queue):
        self.queue=queue
    def set_event(self, event):
        self.event=event
    def set_shm_name(self, shm_name):
        self.shm_name=shm_name
    
    
    
