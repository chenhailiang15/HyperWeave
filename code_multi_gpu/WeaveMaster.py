import time
from NodeCommunicate import NodeMessageSender,NodeMessageReceiver
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
from Job import Job


# if __name__=="__main__":
#     worker_ip="10.26.128.51"
#     worker_port=8000
#     node_message_sender=NodeMessageSender(worker_ip,worker_port)
#     while True:
#         send_info=input("发送数据：")
#         node_message_sender.send(send_info)
        
        





class WeaveMaster:
    
    def __init__(self, print_level=0):
        self.print_level=print_level
        self.clock_time_factor=1000
        self.job_time_factor=1000
        self.schedule_interval=5
        self.schedule_strategy="random"
        self.model_name_list=["AlexNet","ResNet18","ResNet50","VGG16","MobileNetv2"]
        self.batch_size_list=[8,16,32,64,128]
        self.epoch_list=[5,10,15,20]
        self.master_port=2000
        self.worker_port=3000
        
        self.analyze_2080_loader=AnalyzeDataLoader("Analyzer-NVIDIA_GeForce_RTX_2080.csv",print_level)
        self.analyze_2080ti_loader=AnalyzeDataLoader("Analyzer-NVIDIA_GeForce_RTX_2080_Ti.csv",print_level)
        
        self.command_sender=NodeMessageSender(aim_node_ip="10.26.128.51")
        self.info_receiver=NodeMessageReceiver(port=8001)
        
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
        job_name=ali_trace["job_name"]
        model_name=random.choice(self.model_name_list)
        batch_size=random.choice(self.batch_size_list)
        parrallel_num=ali_trace["plan_gpu"]/100 if ali_trace["plan_gpu"]<=400 else 4
        
        model_info=model_name+"-"+str(batch_size)+"-"+str(int(math.ceil(parrallel_num)))
        init_time=self.analyze_2080_loader.get_value(model_info,"stage_init","time")
        epoch_time=self.analyze_2080_loader.get_value(model_info,"stage_sample","time")+self.analyze_2080_loader.get_value(model_info,"stage_train","time")
        cal_epoch=math.ceil((ali_trace["plan_gpu"]/self.job_time_factor-init_time)/epoch_time)
        total_epochs=cal_epoch if cal_epoch>5 else 5
        
        plan_cpu=ali_trace["plan_cpu"]/ali_trace["cpu_usage"]*self.analyze_2080_loader.get_value(model_info,"stage_sample","cpu")
        plan_mem=ali_trace["plan_mem"]/ali_trace["avg_mem"]*self.analyze_2080_loader.get_value(model_info,"stage_sample","mem")
        arrive_time=time.time()
        
        
        job=Job()
        job.set_model_info(job_name, model_name,total_epochs, batch_size)
        job.set_plan_resource(plan_cpu, plan_mem, parrallel_num)
        job.set_arrive_time(arrive_time)
        return job
        
    #调度子线程，间隔schedule_interval（秒）后，执行一次调度。未调度成功的job需要返回，重新放入队列
    #调度停止的条件是job不再到来（self.schedule_flage=False），并且队列为空(qsize==0)
    def schedule_subthreading(self):
        while self.schedule_flage or self.wait_schedule_queue.qsize()>0:
            time.sleep(self.schedule_interval)
            wait_schedule_list=[]
            
            while self.wait_schedule_queue.qsize()>0:
                wait_schedule_list.append(self.wait_schedule_queue.get())
                
            rest_jobs=self.scheduler.do_schedule(wait_schedule_list)
            for job in rest_jobs:
                self.wait_schedule_queue.put(job)
            
    #job到来的函数，持续运行，直到读取的文件中的job结束
    def job_come(self):
        self.wait_schedule_queue=queue.Queue()
        self.schedule_flage=True
        sub_thread_schedule=threading.Thread(target=self.schedule_subthreading,args=())
        sub_thread_schedule.start()
        
        for index in range(len(self.ali_trace_pd)):
            job=self.generate_job(self.ali_trace_pd.iloc[index,:])
            if self.print_level>=2:
                print(f"{job.get_key_job_info()}")
            self.wait_schedule_queue.put(job)
            
            if index+1<len(self.ali_trace_pd):
                time.sleep(self.ali_trace_pd.loc[index+1,"start_time"]-self.ali_trace_pd.loc[index,"start_time"])
                # a=1
        self.schedule_flage=False
        sub_thread_schedule.join()
        return
    
    #将任务发送给worker执行
    def send_job_to_worker(self,job_f):
        self.command_sender(job_f.to_string())
        
    #任务本地执行
    def execute_job_in_master(self, job_f):
        sub_thread=threading.Thread(target=self.run_command,args=(job_f.command,))
        sub_thread.start()

    #任务具体运行
    def run_command(self,command):
        print("master start command:\n", command)
        os.system(command)
    
    
    
    
    
# worker_ip="10.26.128.51"
# worker_port=8000
# node_message_sender=NodeMessageSender(worker_ip,worker_port)
 #可以修改为根据阿里数据集计算得到，后续修改
    
if __name__=="__main__":
    print_level=10
    weave_master=WeaveMaster(print_level)
    weave_master.job_come()

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
    
    
    
    
    
    
         
