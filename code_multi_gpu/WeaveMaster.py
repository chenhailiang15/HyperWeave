import time
from NodeCommunicate import NodeMessageSender
from WeaveAnalyzer import AnalyzeDataLoader
import json
import secrets
import string
import os
import threading
from util import *
import pandas as pd
import random
import math
from WeaveScheduler import WeaveSchedulor
import queue



# if __name__=="__main__":
#     worker_ip="10.26.128.51"
#     worker_port=8000
#     node_message_sender=NodeMessageSender(worker_ip,worker_port)
#     while True:
#         send_info=input("发送数据：")
#         node_message_sender.send(send_info)
        
        





class WeaveMaster:
    
    def __init__(self, print_flage=False):
        self.clock_time_factor=1000
        self.job_time_factor=1000
        self.schedule_interval=5
        self.schedule_strategy="random"
        self.model_name_list=["AlexNet","ResNet18","ResNet50","VGG16","MobileNetv2"]
        self.batch_size_list=[8,16,32,64,128]
        self.epoch_list=[5,10,15,20]
        self.master_port=2000
        self.worker_port=3000
        
        self.analyze_2080_loader=AnalyzeDataLoader("Analyzer-NVIDIA_GeForce_RTX_2080.csv",print_flage)
        self.analyze_2080ti_loader=AnalyzeDataLoader("Analyzer-NVIDIA_GeForce_RTX_2080_Ti.csv",print_flage)
        
        self.command_sender=NodeMessageSender(aim_node_ip="10.26.128.51")
        
        
        self.scheduler=WeaveSchedulor(self, self.command_sender, self.schedule_strategy)
        ali_trace_pd=self.load_csv("ali_trace_job_info.csv",header=0)
        min_start_time=ali_trace_pd["start_time_j"].min()
        ali_trace_pd["start_time"]=(ali_trace_pd["start_time_j"]-min_start_time)/self.clock_time_factor
        self.ali_trace_pd=ali_trace_pd.sort_values(by="start_time")
        
        print(ali_trace_pd)
        
        
    def load_csv(self, file_name,header=None):
        dataset_dir=get_dataset_dir()
        data_pd=pd.read_csv(dataset_dir+"cluster_exp_data"+"/"+file_name,header=header)
        return data_pd

    def generate_job(self, ali_trace):
        model_name=random.choice(self.model_name_list)
        batch_size=random.choice(self.batch_size_list)
        parrallel_num=ali_trace["plan_gpu"]/100 if ali_trace["plan_gpu"]<=400 else 4
        model_info=model_name+"-"+str(batch_size)+"-"+str(int(math.ceil(parrallel_num)))
        init_time=self.analyze_2080_loader.get_value(model_info,"stage_init","time")
        epoch_time=self.analyze_2080_loader.get_value(model_info,"stage_sample","time")+self.analyze_2080_loader.get_value(model_info,"stage_train","time")
        cal_epoch=math.ceil((ali_trace["plan_gpu"]/self.job_time_factor-init_time)/epoch_time)
        epoch=cal_epoch if cal_epoch>5 else 5
        return model_name, epoch, batch_size, parrallel_num
        
        
    def schedule_subthreading(self):
        while self.schedule_flage or self.wait_schedule_queue.qsize()>0:
            time.sleep(self.schedule_interval)
            wait_schedule_list=[]
            
            while self.wait_schedule_queue.qsize()>0:
                wait_schedule_list.append(self.wait_schedule_queue.get())
                
            rest=self.scheduler.do_schedule(wait_schedule_list)
            for job in rest:
                self.wait_schedule_queue.put(job)
            
        
    def run(self):
        self.wait_schedule_queue=queue.Queue()
        self.schedule_flage=True
        sub_thread_schedule=threading.Thread(target=self.schedule_subthreading,args=())
        sub_thread_schedule.start()
        
        for index in range(len(self.ali_trace_pd)):
            model_name, epoch, batch_size, parrallel_num=self.generate_job(self.ali_trace_pd.iloc[index,:])
            print(f"{model_name}, {epoch}, {batch_size}, {parrallel_num}")
            self.wait_schedule_queue.put([time.time(), model_name, epoch, batch_size, parrallel_num])
            
            if index+1<len(self.ali_trace_pd):
                time.sleep(self.ali_trace_pd.loc[index+1,"start_time"]-self.ali_trace_pd.loc[index,"start_time"])
                # a=1
        self.schedule_flage=False
        sub_thread_schedule.join()
        return
    
    
    def run_command(self,command):
        print("master start command:\n", command)
        os.system(command)
    

    # def evok_executor(self,strategy_all):
    #     command_taskfirst=parameter_analyse(strategy_all["task_first"],"master_spec",prior=True)
    #     command_tasksecond=parameter_analyse(strategy_all["task_second"], "master_spec",)
    #     print(command_taskfirst)
    #     print(command_tasksecond)
    #     thread1=threading.Thread(target=run_command,args=(command_taskfirst,))
    #     thread1.start()
    #     thread2=threading.Thread(target=run_command,args=(command_tasksecond,))
    #     thread2.start()
        
        
    #     # os.system(command_taskfirst+" & "+command_tasksecond)
    #     # os.system(command_tasksecond)
    #     return thread1,thread2



    def execute_schedule(self,model_name, epoch, batch, select_gpu_list):
        world_size=len(select_gpu_list)
        master_gpu_id_list=[x for x in select_gpu_list if x <4]
        worker_gpu_id_list=[x-4 for x in select_gpu_list if x >= 4]
        
        nprocs_list=[len(master_gpu_id_list), len(worker_gpu_id_list)]
        gpu_id_list=[master_gpu_id_list,worker_gpu_id_list]
        
        
        is_cross=True if len(master_gpu_id_list)>0 and len(worker_gpu_id_list)>0 else False
        
        if len(master_gpu_id_list)>0:
            
            command=self.generate_command(True, is_cross, world_size, nprocs_list, gpu_id_list, model_name, epoch, batch)
            print(command)
            sub_thread=threading.Thread(target=self.run_command,args=(command,))
            sub_thread.start()
        if len(worker_gpu_id_list)>0:
            command=self.generate_command(False, is_cross, world_size, nprocs_list, gpu_id_list, model_name, epoch, batch)
            self.command_sender.send(command)
            
    def generate_command(self, is_master, is_cross, world_size, nprocs_list, gpu_id_list, model_name, total_epochs, batch_size, prior=False):
        if is_master:
            node_rank=0
            net_card="eno2"
        else:
            node_rank=1
            net_card="eno1"
        
        if is_cross or is_master:
            MASTER_ADDR="10.26.128.115"
            MASTER_PORT=self.master_port
            self.master_port+=1
        elif (not is_cross) and (not is_master):
            MASTER_ADDR="10.26.128.51"
            MASTER_PORT=self.worker_port
            self.worker_port+=1
        
        layer_num=10
        layer_feature=10
        
        worker_num=4
        squad_data_size=1000
        sample_interval=0.1
        nprocs_list=nprocs_list.__str__().replace(" ","")
        gpu_id_list=gpu_id_list.__str__().replace(" ","")
        # max_sync_num=strategy["max_sync_num"]
        # shm_name_list=strategy["shm_name_list"]
        # shm_name_list=f"{shm_name_list}".replace(" ", "")
        
        
        command=f"python WeaveExecutor.py --MASTER_ADDR {MASTER_ADDR} --MASTER_PORT {MASTER_PORT} --net_card {net_card}  --model_name {model_name} --node_rank {node_rank} \
        --world_size {world_size} --nprocs_list {nprocs_list} --gpu_id_list {gpu_id_list} --layer_num {layer_num} --layer_feature {layer_feature} \
        --batch_size {batch_size} --total_epochs {total_epochs} --worker_num {worker_num} --squad_data_size {squad_data_size} \
        --sample_interval {sample_interval}"
        if prior:
            command=command+" --prior"
        return command
    
    
    
    
# worker_ip="10.26.128.51"
# worker_port=8000
# node_message_sender=NodeMessageSender(worker_ip,worker_port)
 #可以修改为根据阿里数据集计算得到，后续修改
    
if __name__=="__main__":
    print_flage=True
    weave_master=WeaveMaster(print_flage)
    weave_master.run()

    # start_time=time.time()
    # strategy_all=get_strategy()
    # # node_message_sender.send(json.dumps(strategy_all))
    # thread11,thread12=execution_local(strategy_all)
    # strategy_all=get_strategy2()
    # thread21,thread22=execution_local(strategy_all)
    
    
    
    # thread11.join()
    # thread12.join()
    # thread21.join()
    # thread22.join()
    # end_time=time.time()
    
    # print(f"total time:{round(end_time-start_time,2)}")
    
    
    
    
    
    
         
