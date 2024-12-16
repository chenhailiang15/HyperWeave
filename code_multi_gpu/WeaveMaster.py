import time
from NodeCommunicate import CommunicateServer
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
from WeaveMonitor import WeaveMonitor
import queue
from Job import Job 
from Node import Node


#cpu, gpu 按照百分比表示需求和剩余，即1个GPU 表示为100
#mem, gmem按照存储单位表示，本平台中使用MB


class WeaveMaster:
    
    def __init__(self, print_level=0):
        self.print_level=print_level
        self.clock_time_factor=10000
        self.job_time_factor=1000
        self.schedule_interval=5
        self.schedule_strategy="over_sharing"
        self.model_name_list=["AlexNet","ResNet18","ResNet50","VGG16","MobileNetv2"]
        self.batch_size_list=[8,16,32,64,128]
        self.epoch_list=[5,10,15,20]
        
        self.master_ip="10.26.128.51"
        # self.master_port=2000
        self.worker_ip="10.26.128.115"
        # self.worker_port=3000
        
        #正在处理的job数量用于控制程序结束
        self.dealing_job_num=0
        #用于socket包去粘包
        self.buffer=""
        self.end_event=threading.Event()
        
        self.max_cross=1
        self.max_gpu_cross=4
        self.overshared_factor=4
        
        self.init_node()
        
        if self.print_level>0:
            print("master load ali trace...")
        self.load_ali_trace("ali_trace_job_info.csv")
        
        if self.print_level>0:
            print("master init analyze loader...")
        self.analyze_2080_loader=AnalyzeDataLoader("Analyzer-NVIDIA_GeForce_RTX_2080.csv",print_level)
        self.analyze_2080ti_loader=AnalyzeDataLoader("Analyzer-NVIDIA_GeForce_RTX_2080_Ti.csv",print_level)
        
        #资源监视器
        if self.print_level>0:
            print("master init monitor...")
        self.monitor=WeaveMonitor(self.nodes, self.print_level)
        
    
        if self.print_level>0:
            print("master init scheduler...")
        self.scheduler=WeaveSchedulor(self, self.schedule_strategy, print_level=self.print_level)
        
        #通讯器，初始化和启动监听。
        if self.print_level>0:
            print("master init communicator...")
        self.communicator=CommunicateServer(print_level=self.print_level)
        self.communicator.start_connect()
        self.communicator.start_listening(self.message_receive)
        
        
    def init_node(self):
        self.node_num=2
        master_node=Node(node_id="worker", ip="10.26.128.51", net_card="eno1", overshared_factor=self.overshared_factor, max_cross_gpu_job_num=self.max_gpu_cross, print_level=self.print_level)
        master_node.set_init_resouce(48*100,125*1024, 3, 11*1024)
        
        worker_node=Node(node_id="master", ip="10.26.128.115", net_card="eno2", overshared_factor=self.overshared_factor, max_cross_gpu_job_num=self.max_gpu_cross, print_level=self.print_level)
        worker_node.set_init_resouce(48*100, 62*1024, 4, 8*1024)
        
        self.nodes=[master_node, worker_node]
        
    def message_receive(self,message):
        
        self.buffer+=message
        buffer_list=self.buffer.split("--end")
        if len(buffer_list)>1:
            for i in range(len(buffer_list)-1):
                
                job=Job()
                job.load_string(buffer_list[i])
                
                if print_level>5:
                    print("master receive back job:\t", job.job_name)
                    
                    
                self.statistic_end_job(job)
                
            self.buffer=buffer_list[len(buffer_list)-1]
            
        

    def load_ali_trace(self,file_name):
        ali_trace_pd=self.load_csv(file_name,header=0)
        min_start_time=ali_trace_pd["start_time_j"].min()
        ali_trace_pd["start_time"]=(ali_trace_pd["start_time_j"]-min_start_time)/self.clock_time_factor
        self.ali_trace_pd=ali_trace_pd.sort_values(by="start_time")

    def load_csv(self, file_name,header=None):
        dataset_dir=get_dataset_dir()
        data_pd=pd.read_csv(dataset_dir+"cluster_exp_data"+"/"+file_name,header=header)
        return data_pd

    def generate_job(self, ali_trace):
        job_name=ali_trace["job_name"]

        model_name=random.choice(self.model_name_list)
        batch_size=random.choice(self.batch_size_list)
        
        plan_gpu=ali_trace["plan_gpu"] if ali_trace["plan_gpu"]<=700 else 700
        
        parrallel_num=math.ceil(min(plan_gpu, 400)/100)
        model_info=model_name+"-"+str(batch_size)+"-"+str(parrallel_num)
        init_time=self.analyze_2080_loader.get_value(model_info,"stage_init","time")
        epoch_time=self.analyze_2080_loader.get_value(model_info,"stage_sample","time")+self.analyze_2080_loader.get_value(model_info,"stage_train","time")
        cal_epoch=math.ceil((ali_trace["duration_s"]/self.job_time_factor-init_time)/epoch_time)
        total_epochs=cal_epoch if cal_epoch<5 else 5
        
        plan_cpu=ali_trace["plan_cpu"]/ali_trace["cpu_usage"]*self.analyze_2080_loader.get_value(model_info,"stage_sample","cpu")
        plan_mem=ali_trace["plan_mem"]/ali_trace["avg_mem"]*self.analyze_2080_loader.get_value(model_info,"stage_sample","mem")
        
        arrive_time=time.time()
        
        job=Job()
        job.set_model_info(job_name, model_name,total_epochs, batch_size)
        job.set_plan_resource(plan_cpu, plan_mem, plan_gpu)
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
                print(f"{job.job_key_info()}")
            self.wait_schedule_queue.put(job)
            
            if index+1<len(self.ali_trace_pd):
                time.sleep(self.ali_trace_pd.loc[index+1,"start_time"]-self.ali_trace_pd.loc[index,"start_time"])
                # a=1
        self.schedule_flage=False
        sub_thread_schedule.join()
        
        
        return
    
    
    def send_job_to_execution(self, job_f):
        if job_f.node_rank==0:
            self.execute_job_in_master(job_f)
        else:
            self.send_job_to_worker(job_f)
    #将任务发送给worker执行
    def send_job_to_worker(self,job_f):
        if self.print_level>5:
            print(f"send job to worker:{job_f.job_name} ...")
            
        #判断是否仅在worker运行，避免跨机器任务重复计数
        if job_f.world_size==job_f.nprocs_list[1]:
            self.dealing_job_num+=1
        self.communicator.send(job_f.to_string()+"--end")
        
    #任务本地执行
    def execute_job_in_master(self, job_f):
        if self.print_level>5:
            print(f"master execute job:{job_f.job_name} ...")
            
        self.dealing_job_num+=1
        
        sub_thread=threading.Thread(target=self.run_command,args=(job_f,job_f.command,))
        sub_thread.start()

    #任务具体运行
    def run_command(self,job, command):
        if self.print_level>5:
            print("master start command:\n", command)
        job.set_start_time(time.time())
        back=os.system(command)
        if back==0:
            job.succeed()
        else:
            job.failed()
            
        job.set_end_time(time.time())
        #这里很重要，对于资源的回收，结果的统计，都在这里进行
        self.statistic_end_job(job)
        print("执行完成后的返回值：",back)
    
    def statistic_end_job(self,job):
        if self.print_level>3:
            print("end a job:", job.job_name)
        
        #回收资源(需要修改，有配对的，在两个都结束后，再释放资源)
        gpu_list=job.gpu_list
        self.monitor.takeback_resource(job, gpu_list, [job.pack_cpu, job.pack_mem, job.pack_gpu, job.pack_gmem])
        
        if job.is_main:
            
            self.dealing_job_num-=1
            if self.dealing_job_num==0:
                self.end_event.set()
        
            
    
    def wait(self):
        self.end_event.wait()
        
    def close(self):
        self.communicator.close()
import random

random.seed(30)

if __name__=="__main__":
    print_level=10
    weave_master=WeaveMaster(print_level)
    # for i in range(10):
    #     mess="aijf"*10
    #     weave_master.communicator.send(mess+"--end")
    weave_master.job_come()
    weave_master.wait()
    weave_master.close()

    print("The whole process end (by master)!")

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
    
    
    
    
    
    
         
