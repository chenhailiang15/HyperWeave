import time
import json
import secrets
import string
import os
import threading
from util import *
import pandas as pd
import random
import math
import queue
from Job import Job 
from Node import Node
import subprocess
from Recorder import Record
from NodeCommunicate import CommunicateServer
from WeaveAnalyzer import AnalyzeDataLoader, AnalyzeTimeLoader
from WeaveScheduler import WeaveSchedulor
from WeaveMonitor import WeaveMonitor

#cpu, gpu 按照百分比表示需求和剩余，即1个GPU 表示为100
#mem, gmem按照存储单位表示，本平台中使用MB


class WeaveMaster:
    
    def __init__(self,stragey, print_level=0):
        
        
        ##################################################《--设置区域--》开始####################################################
        self.master_is_2080=True
        self.single_node_mode=True
        
        self.system="Muri"  #"Muri" or "Normal"
        self.schedule_strategy=stragey # "FIFO", "SRTF"，"SRSF", "BNPF"   Bucket-based non-blocking parallel first

        
        self.weave_sync_mode=True
        #需要最好手动确认
        self.MPS_mode=True 
        
        self.overshared_factor=1   #等于1存在GPU资源不够的情况
             
        self.password=" "      #"sim2024"for sim812 " "for jf
        
        self.print_level=print_level
        self.clock_time_factor=10000
        self.job_time_factor=1
        self.job_ddl_factor=1             #ddl是任务持续时间的job_ddl_factor倍
        
        self.schedule_interval=10
        
        
        self.model_name_list=model_list_g    #
        self.batch_size_dict=model_to_batch_size_g

        

        self.max_cross=1           #最大跨node任务数量
        self.max_gpu_cross=1       #最大跨GPU任务数量（单node）
        
        
        self.single_job_max_plan_cpu=5*100
        self.single_job_max_plan_mem=10*1024
        self.single_job_max_plan_gpu=4*100
        
        self.ali_trace_file_name="ali_trace_job_info.csv"
        self.analyze_file_name="Analyzer-NVIDIA_GeForce_RTX_2080-tim_12_19_16_38_14.csv"
        self.model_time_file_name="Muri_Analyzer-NVIDIA_GeForce_RTX_2080_Ti-tim_12_26_15_24_41.csv"
        ##################################################《--设置区域--》结束####################################################
        

        
        self.queue_length=[]
        self.block_index=[]   #
        #用于socket包去粘包
        self.buffer=""
        self.end_event=threading.Event()
        self.lock=threading.Lock()
        ############统计信息##############
        
        self.job_come_num=0
        self.job_end_num=0   #记录main job
        
        #正在处理的job数量用于控制程序结束
        self.job_dealing_num=0
        
        self.command_start_num=0
        self.command_end_num=0
        
        self.succeed_job_num=0
        self.failed_job_num=0
        
        self.job_wait_time_list=[]
        self.job_complete_time_list=[]
        #################################
        self.init_node()
        self.init_MPS()
        # exit(8)
        if self.print_level>0:
            print("master load ali trace...")
        self.load_ali_trace(self.ali_trace_file_name)
        
        if self.print_level>0:
            print("master init analyze loader...")
        self.analyze_loader=AnalyzeDataLoader(self.analyze_file_name, print_level)
        if self.system=="Muri":
            self.analyze_time_loader=AnalyzeTimeLoader(self.model_time_file_name, print_level)
        # self.analyze_2080ti_loader=AnalyzeDataLoader("Analyzer-NVIDIA_GeForce_RTX_2080_Ti.csv",print_level)
        
        #资源监视器
        if self.print_level>0:
            print("master init monitor...")
        self.monitor=WeaveMonitor(self.nodes, self.print_level)
        
    
        if self.print_level>0:
            print("master init scheduler...")
        self.scheduler=WeaveSchedulor(self, print_level=self.print_level)
        
        #通讯器，初始化和启动监听。
        if self.print_level>0:
            print("master init communicator...")
        if not self.single_node_mode:
            self.communicator=CommunicateServer(print_level=self.print_level)
            self.communicator.start_connect()
            self.communicator.start_listening(self.message_receive)

    #初始化node信息
    def init_node(self):
        
        node_2080=Node(node_id="node_2080", ip="10.26.128.115", net_card="eno2", overshared_factor=self.overshared_factor, max_cross_gpu_job_num=self.max_gpu_cross, print_level=self.print_level)
        node_2080.set_init_resouce(48*100, 60*1024, 4, 6*1024)
        node_2080ti=Node(node_id="node_2080ti", ip="10.26.128.51", net_card="eno1", overshared_factor=self.overshared_factor, max_cross_gpu_job_num=self.max_gpu_cross, print_level=self.print_level)
        node_2080ti.set_init_resouce(48*100,120*1024, 3, 10*1024)
        
        if self.single_node_mode:
            self.node_num=1
            if self.master_is_2080:
                self.nodes=[node_2080]
            else:
                self.nodes=[node_2080ti]
        else:
            self.node_num=2
            if self.master_is_2080:
                self.nodes=[node_2080, node_2080ti]
            else:
                self.nodes=[node_2080ti, node_2080]
            
            
    #初始化MPS
    def init_MPS(self):
        
        
        if self.MPS_mode ==True:
            flage = start_MPS(self.password)
            if flage:
                print("MPS 开启")
                return True
            else:
                print("MPS 开启失败")
                exit(-1)
                return False
        else:
            flage = stop_MPS(self.password)
            if flage:
                print("MPS 关闭")
                return True
            else:
                print("MPS 关闭失败")
                exit(-1)
                return False

    def load_ali_trace(self,file_name):
        ali_trace_pd=self.load_csv(file_name,header=0)
        min_start_time=ali_trace_pd["start_time_j"].min()
        ali_trace_pd["start_time"]=(ali_trace_pd["start_time_j"]-min_start_time)/self.clock_time_factor
        self.ali_trace_pd=ali_trace_pd.sort_values(by="start_time")

    def load_csv(self, file_name,header=None):
        dataset_dir=get_dataset_dir()
        data_pd=pd.read_csv(dataset_dir+"cluster_exp_data"+"/"+file_name,header=header)
        return data_pd

    def generate_job(self, ali_trace,job_idx):
        
        if ali_trace["cpu_usage"]==0 or ali_trace["avg_mem"]==0:
            return None
        job_name=ali_trace["job_name"]
        while True:
            model_name=random.choice(self.model_name_list)
            batch_size=random.choice(self.batch_size_dict[model_name])

            plan_gpu=ali_trace["plan_gpu"] if ali_trace["plan_gpu"]<=self.single_job_max_plan_gpu else self.single_job_max_plan_gpu
            
            
            parrallel_num=math.ceil(min(plan_gpu, 400)/100)
            model_info=model_name+"-"+str(batch_size)+"-"+str(parrallel_num)
            duration_time=ali_trace["duration_s"]/self.job_time_factor
            init_time=self.analyze_loader.get_value(model_info,"stage_init","time")
            epoch_time=self.analyze_loader.get_value(model_info,"stage_sample","time")+self.analyze_loader.get_value(model_info,"stage_train","time")
            model_duration_time=init_time+epoch_time
            
            if model_duration_time<duration_time:
                break
        
        
        total_epochs=math.ceil((ali_trace["duration_s"]/self.job_time_factor-init_time)/epoch_time)
        
        
        plan_cpu=min(ali_trace["plan_cpu"]/ali_trace["cpu_usage"]*self.analyze_loader.get_value(model_info,"stage_sample","cpu"), self.single_job_max_plan_cpu)
        plan_mem=min(ali_trace["plan_mem"]/ali_trace["avg_mem"]*self.analyze_loader.get_value(model_info,"stage_sample","mem"),self.single_job_max_plan_cpu)
        
        arrive_time=time.time()
        
        ddl_time=arrive_time+duration_time*self.job_ddl_factor
        
        job=Job(job_idx,self.system)
        if model_name == "GCN":
            job.set_model_info(job_name, model_name,total_epochs, batch_size, layer_num=100, layer_feature=100)
        else:
            job.set_model_info(job_name, model_name,total_epochs, batch_size)
        
        job.set_plan_resource(plan_cpu, plan_mem, plan_gpu)
        job.set_arrive_time(arrive_time)
        job.set_ddl_time(ddl_time)
        job.set_duration_time(duration_time)
        return job
        
    #调度子线程，间隔schedule_interval（秒）后，执行一次调度。未调度成功的job需要返回，重新放入队列
    #调度停止的条件是job不再到来（self.job_come_flage=False），并且队列为空(qsize==0)
    def schedule_subthreading(self):
        while self.job_come_flage or self.wait_schedule_queue.qsize()>0:
            time.sleep(self.schedule_interval)
            wait_schedule_list=[]
            self.queue_length.append(self.wait_schedule_queue.qsize())
            
            while self.wait_schedule_queue.qsize()>0:
                wait_schedule_list.append(self.wait_schedule_queue.get())
            
            
            rest_jobs=self.scheduler.do_schedule(wait_schedule_list)
            
            for job in rest_jobs:
                # print(f"wait for next scheduling:job name({job.job_name})")
                self.wait_schedule_queue.put(job)
                
            
    #job到来的函数，持续运行，直到读取的文件中的job结束
    def job_come(self):
        self.start_time=time.time()
        
        self.wait_schedule_queue=queue.Queue()
        self.job_come_flage=True
        sub_thread_schedule=threading.Thread(target=self.schedule_subthreading,args=())
        sub_thread_schedule.start()
        
        for index in range(len(self.ali_trace_pd)):
            job=self.generate_job(self.ali_trace_pd.iloc[index,:], self.job_come_num)
            if job ==None: #由于数据原因，可能无法生成Job，因此跳过
                continue
            if self.print_level>=2:
                print(f"job ${job.job_idx}$ come ( detailed info :{job.job_key_info()})")
            self.wait_schedule_queue.put(job)
            self.job_come_num+=1
            if index+1<len(self.ali_trace_pd):
                time.sleep(self.ali_trace_pd.loc[index+1,"start_time"]-self.ali_trace_pd.loc[index,"start_time"])
 
        self.job_come_flage=False
        sub_thread_schedule.join()

        return
    
    
    def send_job_to_execution(self, job_f):
        with self.lock:
            #记录统计数据
            self.command_start_num+=1
            if job_f.is_main:
                self.job_dealing_num+=1
                
            if job_f.node_rank==0:
                if self.print_level>5:
                    print(f"master execute job:{job_f.job_name} ...")
                self.execute_job_in_master(job_f)
            else:
                if self.print_level>5:
                    print(f"send job to worker:{job_f.job_name} ...")
                self.send_job_to_worker(job_f)
                
    #将任务发送给worker执行
    def send_job_to_worker(self,job_f):
        self.communicator.send(job_f.to_string()+"--end")
        
    def message_receive(self,message):
        
        self.buffer+=message
        buffer_list=self.buffer.split("--end")
        if len(buffer_list)>1:
            for i in range(len(buffer_list)-1):
                
                job=Job()
                job.load_string(buffer_list[i])
                
                if self.print_level>5:
                    print(f"master receive back job ${job.job_name}$" )

                self.statistic_end_job(job)
                
            self.buffer=buffer_list[len(buffer_list)-1]
            
    #任务本地执行
    def execute_job_in_master(self, job_f):
        sub_thread=threading.Thread(target=self.run_command,args=(job_f,job_f.command,))
        sub_thread.start()

    #任务具体运行
    def run_command(self,job, command):
        if self.print_level>5:
            print(f"******master start job ${job.job_name}$ with command:\t {command}")
        job.set_start_time(time.time())
        back=os.system(command)
        if back==0:
            job.succeed()
        else:
            job.failed()
            
        job.set_end_time(time.time())
        #这里很重要，对于资源的回收，结果的统计，都在这里进行
        self.statistic_end_job(job)
        print(f"******master end job ${job.job_name}$ with back code: {back}")
    
    def statistic_end_job(self,job):
        
        if self.print_level>3:
            print("end a job:", job.job_idx)
        
        #回收资源(需要修改，有配对的，在两个都结束后，再释放资源)
        if self.system=="Weave":
            self.monitor.takeback_resource(job)
        else:
            self.monitor.takeback_resource(job,plan=True)
            
        with self.lock:    
            #统计信息
            self.command_end_num+=1
            
            if job.is_main:
                
                self.job_end_num+=1
                self.job_dealing_num-=1
                if job.succeed_flage:
                    self.succeed_job_num+=1
                    self.job_wait_time_list.append(job.start_time-job.arrive_time)
                    self.job_complete_time_list.append(job.end_time-job.arrive_time)
                else:
                    self.failed_job_num+=1
                    
            if self.job_come_flage==False and self.job_come_num == self.job_end_num and self.command_start_num == self.command_end_num:
                print("event set")
                self.end_event.set()
                
        print("********************************** current status **********************************")
        print(f"job come number:{self.job_come_num}\tjob end number:{self.job_end_num}\tjob dealing number:{self.job_dealing_num}")
        print(f"job succeed number:{self.succeed_job_num}\tjob failed number:{self.failed_job_num}")
        print(f"command start number:{self.command_start_num}\t command end number:{self.command_end_num}")
        print("************************************************************************************")
            
    def print_job_time_info(self):
        size, _mean, _min, _max, per_50, per_90, per_95=analyze_datas(self.job_wait_time_list)
        out_string1=f"job wait time: size-{size}, \tmean-{_mean}, \tmin-{_min}, \tmax-{_max}, \tpercentile50-{per_50}, \tpercentile90-{per_90}, \tpercentile95-{per_95}"
        print(out_string1)
        size, _mean, _min, _max, per_50, per_90, per_95=analyze_datas(self.job_complete_time_list)
        out_string2=f"job complete time(JCT): size-{size}, \tmean-{_mean}, \tmin-{_min}, \tmax-{_max}, \tpercentile50-{per_50}, \tpercentile90-{per_90}, \tpercentile95-{per_95}"
        print(out_string2)
        
        self.sum_string=out_string1+"\n"+out_string2
    def wait(self):
        
        self.end_event.wait()
        

        
    def close(self):
        self.makespan=time.time()-self.start_time
        if not self.single_node_mode:
            self.communicator.close()
            
    def get_sum_info(self):
        
        
        temp_string=f"****************************************************{self.schedule_strategy}***********************************************************\n"
        temp_string+="    ^^^^    ^^^^    ^^^^    ^^^^    ^^^^    ^^^^    parameters in experiment    ^^^^    ^^^^    ^^^^    ^^^^    ^^^^    ^^^^    \n"
        temp_string+=f"master_is_2080:{self.master_is_2080}, single_node_mode:{self.single_node_mode}\n"
        temp_string+=f"system name:{self.system}, \tMPS:{self.MPS_mode}, \tSync:{self.weave_sync_mode}\n"
        temp_string+=f"overshared_factor:{self.overshared_factor}, \tmax_cross:{self.max_cross}, \tmax_gpu_cross:{self.max_gpu_cross}\n"
        temp_string+=f"single_job_max_plan_cpu:{self.single_job_max_plan_cpu}, \tsingle_job_max_plan_mem:{self.single_job_max_plan_mem}, \tsingle_job_max_plan_gpu:{self.single_job_max_plan_gpu}\n"
        temp_string+=f"clock_time_factor:{self.clock_time_factor}, \tjob_time_factor:{self.job_time_factor}, \tjob_ddl_factor:{self.job_ddl_factor}\n"
        temp_string+=f"model_name_list:{self.model_name_list}\n"
        temp_string+=f"batch_size_dict:{self.batch_size_dict}\n"
        temp_string+=f"schedule_interval:{self.schedule_interval}\n"
        temp_string+=f"ali_trace_file_name:{self.ali_trace_file_name}\n"
        temp_string+=f"analyze_file_name:{self.analyze_file_name}\n"
        temp_string+="    ^^^^    ^^^^    ^^^^    ^^^^    ^^^^    time info in following    ^^^^    ^^^^    ^^^^    ^^^^    ^^^^    \n"
        
        temp_string+=f"all job num:{self.job_come_num}, \tsucceed job num:{self.succeed_job_num}, \tfailed job num:{self.failed_job_num}\n"
        temp_string+=f"makespan:{self.makespan}\n"
        temp_string+=f"queue length{self.queue_length}\n"
        
        return temp_string+self.sum_string+"\n\n\n"
        
import random

random.seed(3)

def Record_resource( gpu_id, out_dir, out_file_name,event):
    record=Record(gpu_id=gpu_id,net_card="", sample_interval=0.1,out_dir=out_dir, out_file_name=out_file_name,event=event,print_flage=False)
    record.run()
    
    
def experiment_all(file, strategy,formatted_time, version):
    
    parent_dir  = os.path.dirname(os.path.abspath(os.curdir))
    resource_file_name="Resource_record_"+version+"_"+strategy+"_"+formatted_time+".csv"
    event=threading.Event()
    subTread_record=threading.Thread(target=Record_resource,args=(-1, parent_dir+"/output/",resource_file_name,event))
    subTread_record.start()
        
    print_level=10
    weave_master=WeaveMaster(strategy, print_level)
    weave_master.job_come()
    weave_master.wait()
    weave_master.print_job_time_info()
    weave_master.close()
    out_string=weave_master.get_sum_info()
    file.write(out_string)
    file.flush()
    
    event.set()
    subTread_record.join()
    print(f"The process end (by master) with {strategy}!")




if __name__=="__main__":
    version="v2.0.0"
    parent_dir  = os.path.dirname(os.path.abspath(os.curdir))
    # 格式化输出
    now_time    = datetime.datetime.now()
    formatted_time = now_time.strftime('%m_%d_%H_%M_%S')
    sum_info_file_name="SumInfo_Weave_"+version+"_"+formatted_time+".txt"
    file=open(parent_dir+"/output/"+sum_info_file_name,"w")
    experiment_all(file, "FIFO",formatted_time, version)
    experiment_all(file, "SRTF",formatted_time, version)
    experiment_all(file, "SRSF",formatted_time, version)
    file.close()
    
    