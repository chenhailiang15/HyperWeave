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
from Job import Job, Instance 
from Node import Node
import subprocess
from Recorder import Record
from NodeCommunicate import CommunicateServer
from WeaveAnalyzer import AnalyzeDataLoader
from WeaveScheduler import WeaveSchedulor
from WeaveMonitor import WeaveMonitor

#cpu, gpu 按照百分比表示需求和剩余，即1个GPU 表示为100
#mem, gmem按照存储单位表示，本平台中使用MB


class WeaveMaster:
    
    def __init__(self,args, file_trace,  print_level=0):
        
        
        ##################################################《--设置区域--》开始####################################################
        self.args=args
        self.system=args.system           #"Muri" or "Normal"
        self.schedule_strategy=args.strategy   # "FIFO", "SRTF"，"SRSF", "BNPF"   Bucket-based Non-blocking SRSF
        self.node_kind=args.node_kind
        
        self.weave_sync_mode=(self.args.sync_flage=="True")
        #需要最好手动确认
        self.MPS_mode=(self.args.mps_flage=="True")
        self.single_node_mode=True    #实验中固定为True
        
        if self.system=="Weave":
            self.overshared_factor=2   
        else:
            self.overshared_factor=1       #等于1存在GPU资源不够的情况
            
        if self.schedule_strategy=="BN-SRSF":
            self.bucket_length=100000
        self.couple_init_iter_percent=0.2
            
        self.file_trace=file_trace
        
        self.password=" "      #"sim2024"for sim812 " "for jf
        
        self.print_level=print_level
        
        self.job_come_time_factor=1
        self.job_duration_time_factor=1
        self.job_ddl_factor=10             #ddl是任务持续时间的job_ddl_factor倍

        self.schedule_interval=10
        self.status_out_interval=10
        
        
        self.model_name_list=model_list_g    #
        self.batch_size_dict=model_to_batch_size_g
        
        self.max_cross=1           #最大跨node任务数量
        self.max_gpu_cross=1       #最大跨GPU任务数量（单node）
        
        self.ali_trace_job_info_file_name="ali_trace_job_info_long.csv"
        self.analyze_file_name="Analyzer-NVIDIA_GeForce_RTX_2080-tim_12_19_16_38_14.csv"
        self.model_time_file_name="Muri_Analyzer-NVIDIA_GeForce_RTX_2080_Ti-tim_12_26_15_24_41.csv"
        if self.args.model_kind=="all_model":
            self.model_info_file_name="Full_model_info_12_27_21_27_41.txt"
        elif self.args.model_kind=="cv_model":
            self.model_info_file_name="CV_model_info_01_13_09_26_50.txt"
        else:
            print("model_kind wrong!")
            exit(-1)
        ##################################################《--设置区域--》结束####################################################
        
        self.queue_length=[]
        self.block_index=[]   #
        self.model_info_list=[]
        self.model_info_list_index=0
        self.model_info_list_max=0
        #用于socket包去粘包
        self.buffer=""
        self.end_event=threading.Event()
        self.lock=threading.Lock()
        ############统计信息##############
        
        self.job_come_num=0
        self.job_not_start_num=0
        self.job_start_num = 0
        self.job_end_num=0   #记录main job
        self.job_dealing_num = 0  # 正在处理的job数量用于控制程序结束

        self.instance_start_num=0
        self.instance_end_num=0
        self.instance_dealing_num=0

        self.command_start_num=0
        self.command_end_num=0
        self.command_dealing_num=0
        
        self.succeed_job_num=0
        self.failed_job_num=0
        
        self.job_wait_time_list=[]
        self.job_complete_time_list=[]
        self.instance_global_idx = 0
        self.wait_schedule_queue=queue.Queue()
        self.write_head=True
        self.spec_gpu_id=args.gpu_id_list
        #################################
        # self.init_MPS()
        
        if self.print_level>0:
            print("load node info...")
        self.init_node()
        
        if self.print_level>0:
            print("load job info...")
        self.load_ali_trace(self.ali_trace_job_info_file_name)
        
        if self.print_level>0:
            print("init analyze loader...")
        self.analyze_loader=AnalyzeDataLoader(self.analyze_file_name, print_level)
        self.analyze_loader.load_time_csv(self.model_time_file_name)
        
        #资源监视器
        if self.print_level>0:
            print("master init monitor...")
        self.monitor=WeaveMonitor(self.nodes, self.print_level)
        
    
        if self.print_level>0:
            print("master init scheduler...")
        self.scheduler=WeaveSchedulor(self, print_level=self.print_level)
        
        if not self.single_node_mode:
            #通讯器，初始化和启动监听。
            if self.print_level>0:
                print("master init communicator...")
            self.communicator=CommunicateServer(print_level=self.print_level)
            self.communicator.start_connect()
            self.communicator.start_listening(self.message_receive)
        self.init_model_info()
        
    #初始化node信息
    def init_node(self):
        self.nodes=[]
        if self.node_kind=="4*3090":
            node_3090=Node(self, 0, "3090node", "10.26.0.4", "eno1", self.overshared_factor, self.print_level)
            node_3090.set_init_resouce(96*100, 250*1024, 4, 24*1024*args.gpu_mem_percent, self.spec_gpu_id)
            self.nodes.append(node_3090)
            self.node_num =1
        
        elif self.node_kind=="s4*3090":
            node_s3090=Node(self, 0, "s3090node", "10.26.128.51", "eno1", self.overshared_factor, self.print_level)
            node_s3090.set_init_resouce(48*100,120*1024, 4, 24*1024*args.gpu_mem_percent, self.spec_gpu_id)
            self.nodes.append(node_s3090)
            self.node_num=1
            
        
        elif self.node_kind=="3*2080ti":
            node_2080ti=Node(self, 0, "2080tinode", "10.26.128.51", "eno1", self.overshared_factor, self.print_level)
            node_2080ti.set_init_resouce(48*100,120*1024, 3, 11*1024*args.gpu_mem_percent, self.spec_gpu_id)
            self.nodes.append(node_2080ti)
            self.node_num=1
            
        elif self.node_kind=="4*2080":
            node_2080=Node(self, 0, "2080node", "10.26.128.115", "eno2", self.overshared_factor, self.print_level)
            node_2080.set_init_resouce(48*100, 60*1024, 4, 8*1024*args.gpu_mem_percent, self.spec_gpu_id)
            self.nodes.append(node_2080)
            self.node_num=1
            
        else:
            print("node kind parameter wrong!")
            exit(-1)

            
            
    #初始化MPS
    def init_MPS(self):
        
        
        if self.MPS_mode ==True:
            flage = start_MPS(self.print_level)
            if flage:
                print("MPS 开启")
                return True
            else:
                print("MPS 开启失败")
                exit(-1)
                return False
        else:
            flage = stop_MPS(self.print_level)
            if flage:
                print("MPS 关闭")
                return True
            else:
                print("MPS 关闭失败")
                exit(-1)
                return False
            
    def init_model_info(self):
        file=open(get_dataset_dir()+"exp_data/"+self.model_info_file_name,"r")
        for line in file.readlines():
            self.model_info_list.append(line[0:-1])
            self.model_info_list_max+=1
        
    def load_ali_trace(self,file_name):
        ali_trace_pd=self.load_csv(file_name,header=0)
        min_start_time=ali_trace_pd["start_time_j"].min()
        ali_trace_pd["start_time"]=(ali_trace_pd["start_time_j"]-min_start_time)/self.job_come_time_factor
        self.ali_trace_pd=ali_trace_pd.sort_values(by="start_time")

    def load_csv(self, file_name,header=None):
        dataset_dir=get_dataset_dir()
        data_pd=pd.read_csv(dataset_dir+"exp_data"+"/"+file_name,header=header)
        return data_pd

    def run(self):
        self.start_time=time.time()
        self.job_come_flage=True
        thread_job_come=threading.Thread(target=self.job_come,args=())
        thread_job_come.start()
        thread_schedule=threading.Thread(target=self.schedule_subthreading,args=())
        thread_schedule.start()
        thread_state=threading.Thread(target=self.print_and_store_current_state,args=())
        thread_state.start()
        
        
        thread_job_come.join()
        thread_schedule.join()
        thread_state.join()
        
    
    #job到来的函数，持续运行，直到读取的文件中的job结束
    def job_come(self):
        
        for index in range(len(self.ali_trace_pd)):
            if self.args.job_num==0:
                self.end_event.set()
                break
            job=self.generate_job(self.ali_trace_pd.iloc[index,:], self.job_come_num)
            if job ==None: #由于数据原因，可能无法生成Job，因此跳过
                continue
            if self.print_level>=2:
                print(f"job ${job.job_idx}$ come ( detailed info :{job.job_key_info()})")
            self.wait_schedule_queue.put(job)
            self.job_come_num+=1
            
            #这里判断job数量是否达到要求，如果达到则退出
            if self.job_come_num>=self.args.job_num:
                break
            
            if index+1<len(self.ali_trace_pd) and args.job_together_flage=="False":
                time.sleep(self.ali_trace_pd.loc[index+1,"start_time"]-self.ali_trace_pd.loc[index,"start_time"])
 
        self.job_come_flage=False
        return
    
    def generate_job(self, ali_trace,job_idx):
        
        if ali_trace["cpu_usage"]==0 or ali_trace["avg_mem"]==0:
            return None
        model_name=self.model_info_list[self.model_info_list_index%self.model_info_list_max].split("-")[0]
        batch_size=int(self.model_info_list[self.model_info_list_index%self.model_info_list_max].split("-")[1])
        self.model_info_list_index+=1

        plan_gpu=ali_trace["plan_gpu"]
        
        # #挑选只有一个GPU的任务。
        # if plan_gpu>100:
        #     return None

        parrallel_num=math.ceil(min(plan_gpu, 400)/100)
        model_info=model_name+"-"+str(batch_size)+"-"+str(parrallel_num)
        duration_time=ali_trace["duration_s"]/self.job_duration_time_factor
        init_time=self.analyze_loader.get_value(model_info,"stage_init","time")
        epoch_time=self.analyze_loader.get_value(model_info,"stage_sample","time")+self.analyze_loader.get_value(model_info,"stage_train","time")
        model_duration_time=init_time+epoch_time

        if model_duration_time>=duration_time:
            if self.print_level>=2:
                print("job generate fail (duration time is too small)")
            return None

        
        
        total_epochs=math.ceil((ali_trace["duration_s"]/self.job_duration_time_factor-init_time)/epoch_time)
        each_batch_time=self.analyze_loader.get_time_value(model_info, 1)+self.analyze_loader.get_time_value(model_info, 2)+self.analyze_loader.get_time_value(model_info, 3)
        batch_num=math.ceil(self.analyze_loader.get_value(model_info,"stage_train","time")/each_batch_time)

        job = Job(job_idx, self.system)
        job_name = ali_trace["job_name"]
        job.set_model_info(job_name, model_name,total_epochs, batch_size)
        if model_name == "GCN":
            job.set_model_info(job_name, model_name,total_epochs, batch_size, layer_num=100, layer_feature=100)
        else:
            job.set_model_info(job_name, model_name,total_epochs, batch_size)
            
        job.batch_num=batch_num    #设置batch numbere
        job.instance_num=1        #多个instance融合为一个
        
        #设置各阶段时间消耗
        job.time_init=self.analyze_loader.get_time_value(model_info,0)
        job.time_init_iter=self.analyze_loader.get_value(model_info,"stage_sample","time")
        job.time_get_data=self.analyze_loader.get_time_value(model_info,1)
        job.time_forward_back=self.analyze_loader.get_time_value(model_info,2)
        job.time_commu=self.analyze_loader.get_time_value(model_info,3)
        job.time_epoch_no_init_iter=batch_num*(job.time_get_data+job.time_forward_back+job.time_commu)-job.time_get_data
        job.init_iter_percent=job.time_init_iter/(job.time_epoch_no_init_iter+job.time_init_iter)
        # 通过batchnum 和 epoch 以及各阶段时间，计算总持续时间
        duration_time =job.time_init+total_epochs*(job.time_epoch_no_init_iter+job.time_init_iter)

        #设置计划资源使用量
        job.set_plan_resource(ali_trace["plan_cpu"], ali_trace["plan_mem"]*1024, ali_trace["plan_gpu"])

        pack_resource=[ali_trace["plan_cpu"], ali_trace["plan_mem"]*1024, ali_trace["plan_gpu"],0]
        
        if self.node_kind=="4*2080" and ali_trace["plan_mem"]>10:
            print("job generate fail (plan mem is too big!)")
            return None
        
        
        if self.monitor.judge_runable_with_resource(pack_resource,job.parallel_num, plan_flage=True, init=True) == False:
            if self.print_level>=2:
                print("job generate fail (plan resource not runable!)")
            return None


        #设置各阶段实际资源使用量[init stage, pre-iteration stage, iteration stage]
        max_cpu_usage=max(self.analyze_loader.get_value(model_info,"stage_init", "cpu"), self.analyze_loader.get_value(model_info,"stage_sample", "cpu"), self.analyze_loader.get_value(model_info,"stage_train", "cpu"))
        if max_cpu_usage>ali_trace["plan_cpu"]:
            if self.print_level>=2:
                print("job generate fail (plan resource less than used (cpu)!)")
            return None
        job.used_resource_cpu = [self.analyze_loader.get_value(model_info,"stage_init", "cpu"),\
                                 self.analyze_loader.get_value(model_info,"stage_sample", "cpu"),\
                                 self.analyze_loader.get_value(model_info,"stage_train", "cpu")]

        max_mem_usage=max(self.analyze_loader.get_value(model_info,"stage_init", "mem"), self.analyze_loader.get_value(model_info,"stage_sample", "mem"), self.analyze_loader.get_value(model_info,"stage_train", "mem"))
        if max_mem_usage>ali_trace["plan_mem"]*1024:
            if self.print_level>=2:
                print("job generate fail (plan resource less than used (mem)!)")
            return None
        job.used_resource_mem = [self.analyze_loader.get_value(model_info,"stage_init", "mem"),\
                                 self.analyze_loader.get_value(model_info,"stage_sample", "mem"),\
                                 self.analyze_loader.get_value(model_info,"stage_train", "mem")]

        max_gpu_usage = max(self.analyze_loader.get_value(model_info, "stage_init", "gpu"), self.analyze_loader.get_value(model_info, "stage_sample", "gpu"), self.analyze_loader.get_value(model_info, "stage_train", "gpu"))
        if max_gpu_usage>ali_trace["plan_gpu"]:
            if self.print_level>=2:
                print("job generate fail (plan resource less than used (gpu)!)")
            return None
        job.used_resource_gpu = [self.analyze_loader.get_value(model_info,"stage_init", "gpu"),\
                                 self.analyze_loader.get_value(model_info,"stage_sample", "gpu"),\
                                 self.analyze_loader.get_value(model_info,"stage_train", "gpu")]

        # max_gmem_usage = max(self.analyze_loader.get_value(model_info, "stage_init", "gmem"), self.analyze_loader.get_value(model_info, "stage_sample", "gmem"), self.analyze_loader.get_value(model_info, "stage_train", "gmem"))
        if model_name=="Bert":
            factor=2
        else:
            factor=1
            
        job.used_resource_gmem = [self.analyze_loader.get_value(model_info,"stage_init", "gmem"),\
                                  self.analyze_loader.get_value(model_info,"stage_sample", "gmem"),\
                                  self.analyze_loader.get_value(model_info,"stage_train", "gmem")]*factor
        pack_resource=[max(job.used_resource_cpu), max(job.used_resource_mem), max(job.used_resource_gpu), max(job.used_resource_gmem)]
        # 并行度为1，实际使用为188，存在问题
        if self.monitor.judge_runable_with_resource(pack_resource, job.parallel_num, plan_flage=False, init=True) == False:
            if self.print_level>=2:
                print(f"job generate fail (used resource not runable)!{pack_resource}")
            return None
        #设置job到达时间
        arrive_time = time.time()
        ddl_time = arrive_time + duration_time * self.job_ddl_factor
        job.set_arrive_time(arrive_time)
        job.set_ddl_time(ddl_time)
        job.set_duration_time(duration_time)

        instance=Instance(job, 0, self.system)
        instance.instance_global_idx=self.instance_global_idx
        self.instance_global_idx+=1
        instance.set_duration_time(duration_time)
        job.instance_list.append(instance)
        if system=="Muri":
            job.set_time_for_muri()
        
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
                
            
    
    
    def send_instance_to_execution(self, instance_t):
        with self.lock:
            #记录统计数据
            self.command_start_num+=1
            self.command_dealing_num += 1
            #判断当前instance是不是job 的第一个instance
            if instance_t.is_main:
                self.instance_start_num += 1
                self.instance_dealing_num+=1

                if instance_t.job.instance_num==len(instance_t.job.instance_list)+1:
                    self.job_start_num += 1
                    self.job_dealing_num += 1
                    
            if instance_t.node_rank==0:
                if self.print_level>5:
                    print(f"master execute job:{instance_t.instance_name} ...")
                self.execute_instance_in_master(instance_t)
            else:
                if self.print_level>5:
                    print(f"send job to worker:{instance_t.instance_name} ...")
                self.send_instance_to_worker(instance_t)
    
    #任务本地执行
    def execute_instance_in_master(self, instance_t):
        sub_thread=threading.Thread(target=self.run_command,args=(instance_t,instance_t.command,))
        sub_thread.start()

    #任务具体运行
    def run_command(self,instance_t, command):
        if self.print_level>5:
            print(f"******master start job ${instance_t.instance_name}$ with command:\t {command}")
        if instance_t.job.start_time==0:
            instance_t.job.start_time=time.time()
        if instance_t.is_main:
            instance_t.job.dealing_instance_num+=1
        back=os.system(command)
        instance_t.job.dealing_instance_num-=1
        if back==0:
            instance_t.job.succeed_instance_num+=1
            instance_t.succeed()
        else:
            instance_t.job.failed_instance_num+=1
            instance_t.failed()
            
        instance_t.job.set_end_time(time.time())
        #这里很重要，对于资源的回收，结果的统计，都在这里进行
        self.statistic_end_instance(instance_t)
        print(f"******master end instance ${instance_t.instance_name}$ with back code: {back}")
        
    #将任务发送给worker执行
    def send_instance_to_worker(self,instance_t):
        self.communicator.send(instance_t.to_string()+"--end")
        
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
            
    
    
    def statistic_end_instance(self,instance):
        if self.print_level>=2:
            print(f"time: {round(time.time(), 1)} end a sub-instance:", instance.instance_name)

        with self.lock:

            #统计信息
            self.command_end_num += 1
            self.command_dealing_num -= 1
            if instance.is_main:
                self.instance_end_num+=1
                self.instance_dealing_num-=1
                # 回收资源
                if self.system == "Weave":
                    self.monitor.takeback_resource(instance)
                else:
                    self.monitor.takeback_resource(instance, plan=True)

                if instance.job.succeed_instance_num+instance.job.failed_instance_num == instance.job.instance_num:
                    self.job_end_num+=1
                    self.job_dealing_num-=1
                    if instance.job.succeed_instance_num == instance.job.instance_num:
                        self.succeed_job_num+=1
                        self.job_wait_time_list.append(instance.job.start_time-instance.job.arrive_time)
                        self.job_complete_time_list.append(time.time()-instance.job.arrive_time)
                    else:
                        self.failed_job_num+=1

            #job不再到来，并且到来的job数量等于结束的job数量。此时仿真结束
            if self.job_come_flage==False and self.job_come_num == self.job_end_num:
                self.end_event.set()    #停止out info
                # self.set_makespan()     #统计系统运行时间
            
    def print_and_store_current_state(self):
        self.print_current_state()
        self.write_current_state()

        time.sleep(self.status_out_interval)
        if not self.end_event.is_set():
            self.print_and_store_current_state()
        else:
            self.print_current_state()
            self.write_current_state()
            
    def print_current_state(self):
        # self.update_job_not_start_num()
        self.job_not_start_num=self.wait_schedule_queue.qsize()
        # assert self.job_come_num==self.job_not_start_num+self.job_start_num
        if self.print_level>=1:
            out_string=f"************************current status (now:{time.time()-self.start_time}) *******************************\n"
            out_string+=f"job come number:{self.job_come_num}\tjob not start number (in queue):{self.job_not_start_num}\n"
            out_string+=f"job start number:{self.job_start_num}\tjob dealing number:{self.job_dealing_num}\tjob end number:{self.job_end_num}(S:{self.succeed_job_num}/F:{self.failed_job_num})\n"
            out_string+=f"instance start number:{self.instance_start_num}\tinstance dealing number:{self.instance_dealing_num}\tinstance end number:{self.instance_end_num}\n"
            out_string+=f"command start number:{self.command_start_num}\tcommand dealing number:{self.command_dealing_num}\tcommand end number:{self.command_end_num}\n\n"
            print(out_string)
            
    def write_current_state(self):
        if self.file_trace!=None:
            if self.write_head == True:
                self.file_trace.write("ave_cpu_allocate, ave_mem_allocate, ave_gpu_allocate, ave_gmem_allocate, idel_gpu_num, job_not_start_num, job_dealing_num\n")
                self.write_head=False
                
            idel_gpu_num=0
            ave_cpu_allocate=0
            ave_mem_allocate=0
            ave_gpu_allocate=0
            ave_gmem_allocate=0

            for node_index in range(self.node_num):
                idel_gpu_num+=self.nodes[node_index].get_idle_gpu_num()
                [ave_cpu, ave_mem, ave_gpu, ave_gmem]=self.nodes[node_index].get_ave_allocate_resource()
                ave_cpu_allocate+=ave_cpu
                ave_mem_allocate += ave_mem
                ave_gpu_allocate += ave_gpu
                ave_gmem_allocate += ave_gmem
            ave_cpu_allocate=ave_cpu_allocate/self.node_num
            ave_mem_allocate=ave_mem_allocate/self.node_num
            ave_gpu_allocate=ave_gpu_allocate/self.node_num
            ave_gmem_allocate=ave_gmem_allocate/self.node_num

            out_string=f"{ave_cpu_allocate},{ave_mem_allocate},{ave_gpu_allocate},{ave_gmem_allocate},{idel_gpu_num},{self.job_not_start_num},{self.job_dealing_num}\n "
            self.file_trace.write(out_string)
            self.file_trace.flush()
            
    def update_job_not_start_num(self):
        job_waiting_list=list(self.wait_schedule_queue.queue)
        self.job_not_start_num=0
        for job in job_waiting_list:
            if job.instance_num==len(job.instance_list):
                self.job_not_start_num+=1
        return
                            
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
        
        temp_string=f"system:{self.system}\nschedule_strategy:{self.schedule_strategy}\n"
        temp_string+=f"node_kind:{self.node_kind}\n"
        temp_string+=f"model_kind:{self.args.model_kind}\n"
        temp_string+=f"MPS:{self.MPS_mode}\nSync:{self.weave_sync_mode}\n"
        temp_string+=f"overshared_factor:{self.overshared_factor}\ngpu_mem_percent:{self.args.gpu_mem_percent}\n"
        temp_string+=f"job_come_time_factor:{self.job_come_time_factor}\job_duration_time_factor:{self.job_duration_time_factor}\njob_ddl_factor:{self.job_ddl_factor}\n"
        temp_string+=f"job_num:{self.args.job_num}\n"
        temp_string+=f"schedule_interval:{self.schedule_interval}\n"
        temp_string+=f"makespan_real:{self.makespan}\n"
        temp_string+=f"job_come_num:{self.job_come_num}\nsucceed_job_num:{self.succeed_job_num}\nfailed_job_num:{self.failed_job_num}\n"
        temp_string+=f"queue:{self.queue_length}\n"
        temp_string+=f"job_wait_time_list:{self.job_wait_time_list}\n"
        temp_string+=f"job_complete_time_list:{self.job_complete_time_list}\n"
        temp_string+=f"ali_trace_job_info_file_name:{self.ali_trace_job_info_file_name}\n"
        temp_string+=f"model_info_file_name:{self.model_info_file_name}\n"
        temp_string+=f"analyze_file_name:{self.analyze_file_name}\n"
        temp_string+=f"model_time_file_name:{self.model_time_file_name}\n"
        
        return temp_string+self.sum_string+"\n\n\n"
        
import random

random.seed(3)

def Record_resource( gpu_id_list, out_dir, out_file_name,event):
    record=Record(gpu_id_list, sample_interval=0.1,out_dir=out_dir, out_file_name=out_file_name,event=event,print_flage=False)
    record.run()
    
    
def experiment_one_group_parameters(args, file_sum, file_trace, version, print_level):
    
    cur_dir=os.path.dirname(os.path.abspath(__file__))
    parent_dir  = os.path.dirname(os.path.abspath(cur_dir))
    resource_file_name="Cluster_resource_record_"+args.system+"_"+args.strategy+"-"+version+"_"+formatted_time+".csv"
    event=threading.Event()
    subTread_record=threading.Thread(target=Record_resource,args=(args.gpu_id_list, parent_dir+"/output/",resource_file_name,event))
    subTread_record.start()
        
    weave_master=WeaveMaster(args, file_trace, print_level)
    weave_master.run()
    weave_master.wait()
    weave_master.close()
    
    event.set()
    subTread_record.join()
    if print_level>=1:
        weave_master.print_job_time_info()
        print(f"The cluster end (system:{args.system}, strategy:{args.strategy}, file_sum:{file_sum != None}, file_trace:{file_trace != None}) !")
    
    if file_sum != None:
        out_string=weave_master.get_sum_info()
        file_sum.write(out_string)
        file_sum.flush()




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Prototype platform for DL training job')
    parser.add_argument("--system",default="Muri",type=str)
    parser.add_argument("--strategy", default="FIFO", type=str)
    parser.add_argument("--mps_flage", default="True", type=str)
    parser.add_argument("--sync_flage", default="True", type=str)
    parser.add_argument("--job_together_flage", default="True", type=str)
    parser.add_argument("--node_kind", default="4*3090", type=str, help="s4*3090,4*3090, 3*2080ti, 4*2080")
    parser.add_argument("--model_kind", default="all_model", type=str, help="cv_model, all_model")
    parser.add_argument("--gpu_mem_percent", default=0.9, type=float, help="because of GPU fragement")
    parser.add_argument("--job_num", default=100, type=int)
    parser.add_argument("--gpu_id_list", default=[4,5,6,7], type=parse_list_arg)
    
    parser.add_argument("--print_level", default=11, type=int)
    parser.add_argument("--write_sum", action='store_true')
    parser.add_argument("--write_trace", action='store_true')
    
    args=parser.parse_args()
    
    #判断所给值是否符合要求
    if (args.sync_flage !="True" and  args.sync_flage !="False") or (args.mps_flage !="True" and  args.mps_flage !="False") or\
        (args.job_together_flage !="True" and  args.job_together_flage !="False"):
        print("sync_flage or mps_flage value or job_together_flage wrong!")
        exit(-1)
        
    version="v2.0.0"
    system=args.system
    strategy=args.strategy
    write_sum = (args.write_sum)
    write_trace = args.write_trace
    print_level=args.print_level

    cur_dir=os.path.dirname(os.path.abspath(__file__))
    parent_dir= os.path.dirname(os.path.abspath(cur_dir))
    # 格式化输出
    now_time= datetime.datetime.now()
    formatted_time = now_time.strftime('%m_%d_%H_%M_%S')
    sim_sum_file_name="Cluster_sum-"+system+"_"+strategy+"-"+version+"_"+formatted_time+".txt"
    sim_trace_file_name="Cluster_statistic_trace_"+system+"_"+strategy+"_"+version+"_"+formatted_time+".csv"
    if write_sum:
        file_sum=open(parent_dir+"/output/"+sim_sum_file_name,"w")
    else:
        file_sum=None

    if write_trace:
        file_trace = open(parent_dir + "/output/" + sim_trace_file_name, "w")
    else:
        file_trace=None

    experiment_one_group_parameters(args, file_sum, file_trace, version, print_level)

    if write_sum:
        file_sum.close()
    if write_trace:
        file_trace.close()
    
    