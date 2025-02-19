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
from WeaveAnalyzer import AnalyzeDataLoader
from WeaveScheduler import WeaveSchedulor
from WeaveMonitor import WeaveMonitor
import simpy
import random
import copy
random.seed(3)
result_dict={}
#cpu, gpu 按照百分比表示需求和剩余，即1个GPU 表示为100
#mem, gmem按照存储单位表示，本平台中使用MB

####################已有的问题#####################
#Weave 和 Muri的时间分析结果不一样



#################################################
class WeaveMaster:
    
    def __init__(self,args, file_trace, print_level=0):
        
        
        ##################################################《--设置区域--》开始####################################################
        self.args=args
        self.system=args.system           #"Muri" or "Normal"
        self.schedule_strategy=args.strategy   # "FIFO", "SRTF"，"SRSF", "BNPF"   Bucket-based Non-blocking SRSF
        self.node_kind=args.node_kind
        
        self.weave_sync_mode=(self.args.sync_flage=="True")
        #需要最好手动确认
        self.MPS_mode=(self.args.mps_flage=="True")
        
        if self.system=="Weave":
            self.overshared_factor=args.overshared_factor
        else:
            self.overshared_factor = 1  # 等于1存在GPU资源不够的情况

        if self.schedule_strategy=="BN-SRSF":
            self.bucket_length=args.bucket_length
        self.couple_init_iter_percent=args.couple_init_iter_percent

        self.file_trace=file_trace
        self.job_duration_time_factor = 1
        self.job_ddl_factor = 10  # ddl是任务持续时间的job_ddl_factor倍
        

        self.print_level=print_level

        self.job_come_time_factor = 1
        
        
        
        # self.model_name_list=model_list_g    #
        # self.batch_size_dict=model_to_batch_size_g

        # self.max_cross=1           #最大跨node任务数量
        # self.max_gpu_cross=1       #最大跨GPU任务数量（单node）
        self.ali_trace_node_info_file_name="ali_trace_machine_info.csv"
        if self.args.validation=="True":
            #用于验证系统准确性
            self.ali_trace_job_info_file_name="ali_trace_job_info_sift_plan_gpu.csv"
            self.schedule_interval = 10
            self.status_out_interval=10

        elif self.args.validation=="False":
            #纯仿真
            self.ali_trace_job_info_file_name="ali_trace_job_info_clear.csv"
            self.schedule_interval = 360
            self.status_out_interval=360
            
        else:
            print("validation is wrong!")
            exit(-1)
        
        self.Bigstageresource_file_name="Analyzer-NVIDIA_GeForce_RTX_3090-tim_02_09_23_52_05.csv"
        self.Ministagetime_file_name="Muri_Analyzer-NVIDIA_GeForce_RTX_3090-tim_02_10_07_04_44.csv"
            
        if self.args.model_kind=="all_model":
            self.model_info_file_name="Full_model_info_12_27_21_27_41.txt"
        elif self.args.model_kind=="cv_model":
            self.model_info_file_name="CV_model_info_01_13_09_26_50.txt"
        else:
            print("model_kind wrong!")
            exit(-1)
        ##################################################《--设置区域--》结束####################################################
        

        self.env=simpy.Environment()
        self.queue_length=[]
        self.block_index=[]   #
        self.model_info_list=[]
        self.model_info_list_index=0
        self.model_info_list_max=0
        #用于socket包去粘包
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
        self.wait_schedule_queue = queue.Queue()
        self.write_head=True
        
        self.should_schedule=True
        #################################

        if self.print_level>0:
            print("load ali trace (node info)...")
        self.init_node(self.ali_trace_node_info_file_name)

        if self.print_level>0:
            print("load ali trace (job info)...")
        self.load_ali_trace(self.ali_trace_job_info_file_name)
        
        if self.print_level>0:
            print("init analyze loader...")
        self.analyze_loader=AnalyzeDataLoader(self.Bigstageresource_file_name, print_level)
        self.analyze_loader.load_time_csv(self.Ministagetime_file_name)


        #资源监视器
        if self.print_level>0:
            print("init monitor...")
        self.monitor=WeaveMonitor(self.nodes, self.print_level)

        if self.print_level>0:
            print("init scheduler...")
        self.scheduler=WeaveSchedulor(self, print_level=self.print_level)

        self.init_model_info()


    #初始化node信息
    def init_node(self, file_name):
        
        self.nodes=[]
        if self.args.system=="Muri":
            self.Muri_resource_factor=10
        else:
            self.Muri_resource_factor=1
            
        if self.node_kind=="cluster":
            node_info_pd = self.load_csv(file_name,header=0)
            node_idx_order=0
            for index in range(len(node_info_pd)):
                name=node_info_pd.loc[index, "machine"]
                cpu_cap= node_info_pd.loc[index, "cap_cpu"]*100.0
                mem_cap=node_info_pd.loc[index, "cap_mem"]*1024.0
                gpu_cap=node_info_pd.loc[index, "cap_gpu"]
                if gpu_cap==0:
                    continue
                gmem_cap=gpu_mem_dict[node_info_pd.loc[index, "gpu_type"]]*1024.0
                node=Node(self, self.env,name,node_idx_order, self.overshared_factor, self.print_level)
                node.set_init_resouce(cpu_cap, mem_cap, gpu_cap, gmem_cap)
                self.nodes.append(node)
                node_idx_order += 1
                if node_idx_order>=self.args.node_num:
                    break
            self.node_num = node_idx_order
            
        elif self.node_kind=="4*3090":
            node_3090=Node(self,self.env, "3090node", 0, self.overshared_factor, self.print_level)
            node_3090.set_init_resouce(96*100, 250*1024, 4, 24*1024*self.args.gpu_mem_percent)
            self.nodes.append(node_3090)
            self.node_num =1
            
        elif self.node_kind=="3*2080ti":
            node_2080ti=Node(self, self.env, "2080tinode",0, self.overshared_factor, self.print_level)
            node_2080ti.set_init_resouce(48*100,120*1024, 3, 11*1024*self.args.gpu_mem_percent)
            self.nodes.append(node_2080ti)
            self.node_num=1
            
        elif self.node_kind=="4*2080":
            node_2080=Node(self, self.env, "2080node", 0, self.overshared_factor, self.print_level)
            node_2080.set_init_resouce(48*100, 60*1024, 4, 8*1024*args.gpu_mem_percent)
            self.nodes.append(node_2080)
            self.node_num=1
            
        else:
            print("node kind parameter wrong!")
            exit(-1)
            
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
        self.start_time_real = time.time()
        self.start_time_sim = self.env.now

        self.env.process(self.job_come())
        self.env.process(self.schedule())
        self.env.process(self.print_and_store_current_state())
        self.env.run()

    # job到来的函数，持续运行，直到读取的文件中的job结束
    def job_come(self):
        
        self.job_come_flage = True

        for index in range(len(self.ali_trace_pd)):
            if self.args.job_num==0:
                self.end_event.set()
                break
            job = self.generate_job(self.ali_trace_pd.iloc[index, :], self.job_come_num)
            if job == None:  # 由于数据原因，可能无法生成Job，因此跳过
                continue

            if self.print_level >= 2:
                print(f"time: {self.env.now}\tjob {job.job_idx} \tcome ( detailed info :{job.job_key_info()})")
            self.wait_schedule_queue.put(job)
            self.job_come_num += 1
            self.job_not_start_num+=1
            self.should_schedule=True  #有来的，则进行调度
            if self.job_come_num>=self.args.job_num:
                break

            if index + 1 < len(self.ali_trace_pd) and self.args.job_together_flage=="False":
                yield self.env.timeout(self.ali_trace_pd.loc[index + 1, "start_time"] - self.ali_trace_pd.loc[index, "start_time"])

        self.job_come_flage = False

        return

    def generate_job(self, ali_trace,job_idx):
        
        if ali_trace["cpu_usage"]==0 or ali_trace["avg_mem"]==0:
            return None

        model_name=self.model_info_list[self.model_info_list_index%self.model_info_list_max].split("-")[0]
        batch_size=int(self.model_info_list[self.model_info_list_index%self.model_info_list_max].split("-")[1])
        self.model_info_list_index+=1

        plan_gpu=ali_trace["plan_gpu"]

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

        if self.args.validation=="True":  #两种模式加载的模型其实际使用资源不同，必须分开
            
            cpu_ratio=self.nodes[0].cpu/self.Muri_resource_factor/100
            #设置各阶段实际资源使用量[init stage, pre-iteration stage, iteration stage]
            max_cpu_usage=cpu_ratio*max(self.analyze_loader.get_value(model_info,"stage_init", "cpu"), self.analyze_loader.get_value(model_info,"stage_sample", "cpu"), self.analyze_loader.get_value(model_info,"stage_train", "cpu"))
            if max_cpu_usage>ali_trace["plan_cpu"]:
                if self.print_level>=2:
                    print("job generate fail (plan resource less than used (cpu)!)")
                return None
            job.used_resource_cpu = [cpu_ratio*self.analyze_loader.get_value(model_info,"stage_init", "cpu"),\
                                    cpu_ratio*self.analyze_loader.get_value(model_info,"stage_sample", "cpu"),\
                                    cpu_ratio*self.analyze_loader.get_value(model_info,"stage_train", "cpu")]

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
        else:
            #设置各阶段实际资源使用量[init stage, pre-iteration stage, iteration stage]
            max_cpu_usage=max(self.analyze_loader.get_value(model_info,"stage_init", "cpu"), self.analyze_loader.get_value(model_info,"stage_sample", "cpu"), self.analyze_loader.get_value(model_info,"stage_train", "cpu"))
            job.used_resource_cpu = [ali_trace["cpu_usage"]*self.analyze_loader.get_value(model_info,"stage_init", "cpu")/max_cpu_usage,\
                                    ali_trace["cpu_usage"]*self.analyze_loader.get_value(model_info,"stage_sample", "cpu")/max_cpu_usage,\
                                    ali_trace["cpu_usage"]*self.analyze_loader.get_value(model_info,"stage_train", "cpu")/max_cpu_usage]

            max_mem_usage=max(self.analyze_loader.get_value(model_info,"stage_init", "mem"), self.analyze_loader.get_value(model_info,"stage_sample", "mem"), self.analyze_loader.get_value(model_info,"stage_train", "mem"))
            job.used_resource_mem = [1024*ali_trace["avg_mem"]*self.analyze_loader.get_value(model_info,"stage_init", "mem")/max_mem_usage,\
                                    1024*ali_trace["avg_mem"]*self.analyze_loader.get_value(model_info,"stage_sample", "mem")/max_mem_usage,\
                                    1024*ali_trace["avg_mem"]*self.analyze_loader.get_value(model_info,"stage_train", "mem")/max_mem_usage]

            max_gpu_usage = max(self.analyze_loader.get_value(model_info, "stage_init", "gpu"), self.analyze_loader.get_value(model_info, "stage_sample", "gpu"), self.analyze_loader.get_value(model_info, "stage_train", "gpu"))
            job.used_resource_gpu = [ali_trace["gpu_wrk_util"]*self.analyze_loader.get_value(model_info,"stage_init", "gpu")/max_gpu_usage,\
                                    ali_trace["gpu_wrk_util"]*self.analyze_loader.get_value(model_info,"stage_sample", "gpu")/max_gpu_usage,\
                                    ali_trace["gpu_wrk_util"]*self.analyze_loader.get_value(model_info,"stage_train", "gpu")/max_gpu_usage]

            max_gmem_usage = max(self.analyze_loader.get_value(model_info, "stage_init", "gmem"), self.analyze_loader.get_value(model_info, "stage_sample", "gmem"), self.analyze_loader.get_value(model_info, "stage_train", "gmem"))
            job.used_resource_gmem = [1024*ali_trace["avg_gpu_wrk_mem"]*self.analyze_loader.get_value(model_info,"stage_init", "gmem")/max_gmem_usage,\
                                    1024*ali_trace["avg_gpu_wrk_mem"]*self.analyze_loader.get_value(model_info,"stage_sample", "gmem")/max_gmem_usage,\
                                    1024*ali_trace["avg_gpu_wrk_mem"]*self.analyze_loader.get_value(model_info,"stage_train", "gmem")/max_gmem_usage]
        pack_resource=[max(job.used_resource_cpu), max(job.used_resource_mem), max(job.used_resource_gpu), max(job.used_resource_gmem)]
        # 并行度为1，实际使用为188，存在问题
        if self.monitor.judge_runable_with_resource(pack_resource, job.parallel_num, plan_flage=False, init=True) == False:
            if self.print_level>=2:
                print(f"job generate fail (used resource not runable)!{pack_resource}")
            return None
        #设置job到达时间
        arrive_time = self.env.now
        ddl_time = arrive_time + duration_time * self.job_ddl_factor
        job.set_arrive_time(arrive_time)
        job.set_ddl_time(ddl_time)
        job.set_duration_time(duration_time)

        instance=Instance(job, 0, self.system)
        instance.instance_global_idx=self.instance_global_idx
        self.instance_global_idx+=1
        instance.set_duration_time(duration_time)
        job.instance_list.append(instance)
        if self.system=="Muri":
            job.set_time_for_muri()
        return job
        
    #调度子线程，间隔schedule_interval（秒）后，执行一次调度。未调度成功的job需要返回，重新放入队列
    #调度停止的条件是job不再到来（self.job_come_flage=False），并且队列为空(qsize==0)
    def schedule(self):
        yield self.env.timeout(self.schedule_interval)
        if self.job_come_flage or self.wait_schedule_queue.qsize()>0:
            
            # self.queue_length.append(self.wait_schedule_queue.qsize())
            if self.should_schedule==True:
                self.should_schedule=False
                wait_schedule_list=[]
                while self.wait_schedule_queue.qsize()>0:
                    wait_schedule_list.append(self.wait_schedule_queue.get())
                
                
                rest_jobs=self.scheduler.do_schedule(wait_schedule_list)
                

                for job in rest_jobs:
                    # print(f"wait for next scheduling:job name({job.job_name})")
                    self.wait_schedule_queue.put(job)
            
            self.queue_length.append(self.wait_schedule_queue.qsize())
            self.env.process(self.schedule())
                
            

    
    
    def send_instance_to_execution(self, instance_t):
        with self.lock:
            self.should_schedule=True
            #记录统计数据
            self.command_start_num += 1
            self.command_dealing_num += 1
            #判断当前instance是不是job 的第一个instance
            if instance_t.is_main:
                self.instance_start_num += 1
                self.instance_dealing_num+=1

                if instance_t.job.instance_num==len(instance_t.job.instance_list)+1:
                    self.job_start_num += 1
                    self.job_dealing_num += 1
                    self.job_not_start_num-=1
            if self.print_level>=2:
                print(f"time: {round(self.env.now, 1)} start a sub-instance:", instance_t.instance_name)
            self.nodes[instance_t.node_rank].execute_instance(instance_t)

    
    def statistic_end_instance(self,instance):
        if self.print_level>=2:
            print(f"time: {round(self.env.now, 1)} end a sub-instance:", instance.instance_name)

        with self.lock:
            self.should_schedule=True
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

                if instance.job.succeed_instance_num+instance.job.failed_instance_num== instance.job.instance_num:
                    self.job_end_num+=1
                    self.job_dealing_num-=1
                    if instance.job.succeed_instance_num == instance.job.instance_num:
                        self.succeed_job_num+=1
                        self.job_wait_time_list.append(instance.job.start_time-instance.job.arrive_time)
                        self.job_complete_time_list.append(self.env.now-instance.job.arrive_time)
                    else:
                        self.failed_job_num+=1

            #job不再到来，并且到来的job数量等于结束的job数量。此时仿真结束
            if self.job_come_flage==False and self.job_come_num == self.job_end_num:
                self.end_event.set()    #停止out info
                self.set_makespan()     #统计系统运行时间

            # self.print_current_state()

    def print_and_store_current_state(self):
        self.print_current_state()
        self.write_current_state()

        yield self.env.timeout(self.status_out_interval)
        if not self.end_event.is_set():
            self.env.process(self.print_and_store_current_state())
        else:
            self.print_current_state()
            self.write_current_state()


    def print_current_state(self):
        # self.update_job_not_start_num()
        # self.job_not_start_num=self.wait_schedule_queue.qsize()
        # assert self.job_come_num==self.job_not_start_num+self.job_start_num
        if self.print_level>=1:
            out_string=f"************************current status (now:{self.env.now}) *******************************\n"
            out_string+=f"job come number:{self.job_come_num}\tjob not start number (in queue):{self.job_not_start_num}\n"
            out_string+=f"job start number:{self.job_start_num}\tjob dealing number:{self.job_dealing_num}\tjob end number:{self.job_end_num}(S:{self.succeed_job_num}/F:{self.failed_job_num})\n"
            out_string+=f"instance start number:{self.instance_start_num}\tinstance dealing number:{self.instance_dealing_num}\tinstance end number:{self.instance_end_num}\n"
            out_string+=f"command start number:{self.command_start_num}\tcommand dealing number:{self.command_dealing_num}\tcommand end number:{self.command_end_num}\n\n"
            print(out_string)

    def write_current_state(self):
        if self.file_trace!=None:
            if self.env.now == 0:
                self.file_trace.write("ave_cpu_allocate,ave_mem_allocate,ave_gpu_allocate,ave_gmem_allocate,idel_gpu_num,job_not_start_num,job_dealing_num,self.job_come_num,self.job_end_num\n")

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

            out_string=f"{ave_cpu_allocate},{ave_mem_allocate},{ave_gpu_allocate},{ave_gmem_allocate},{idel_gpu_num},{self.job_not_start_num},{self.job_dealing_num},{self.job_come_num},{self.job_end_num}\n"
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
        if self.print_level>0:
            print(out_string1)
        size, _mean, _min, _max, per_50, per_90, per_95=analyze_datas(self.job_complete_time_list)
        out_string2=f"job complete time(JCT): size-{size}, \tmean-{_mean}, \tmin-{_min}, \tmax-{_max}, \tpercentile50-{per_50}, \tpercentile90-{per_90}, \tpercentile95-{per_95}"
        if self.print_level>0:
            print(out_string2)
        
        self.sum_string=out_string1+"\n"+out_string2
        result_dict["JCT"]=_mean




    def set_makespan(self):
        self.makespan_sim=self.env.now-self.start_time_sim
        self.makespan_real=time.time()-self.start_time_real

            
    def get_sum_info(self):
        
        #cluster这部分还没修改
        
        temp_string=f"system:{self.system}\nschedule_strategy:{self.schedule_strategy}\n"
        temp_string+=f"node_kind:{self.node_kind}\nvalidation:{self.args.validation}\n"
        temp_string+=f"model_kind:{self.args.model_kind}\n"
        temp_string+=f"MPS:{self.MPS_mode}\nSync:{self.weave_sync_mode}\n"
        temp_string+=f"overshared_factor:{self.overshared_factor}\ngpu_mem_percent:{self.args.gpu_mem_percent}\n"
        temp_string+=f"job_come_time_factor:{self.job_come_time_factor}\njob_duration_time_factor:{self.job_duration_time_factor}\njob_ddl_factor:{self.job_ddl_factor}\n"
        temp_string+=f"node_num:{self.args.node_num}\n"
        temp_string+=f"job_num:{self.args.job_num}\n"
        temp_string+=f"schedule_interval:{self.schedule_interval}\n"
        temp_string+=f"makespan_real:{self.makespan_real}\nmakespan:{self.makespan_sim}\n"
        temp_string+=f"job_come_num:{self.job_come_num}\nsucceed_job_num:{self.succeed_job_num}\nfailed_job_num:{self.failed_job_num}\n"
        temp_string+=f"queue:{self.queue_length}\n"
        temp_string+=f"job_wait_time_list:{self.job_wait_time_list}\n"
        temp_string+=f"job_complete_time_list:{self.job_complete_time_list}\n"
        temp_string+=f"ali_trace_job_info_file_name:{self.ali_trace_job_info_file_name}\n"
        temp_string+=f"ali_trace_node_info_file_name:{self.ali_trace_node_info_file_name}\n"
        temp_string+=f"model_info_file_name:{self.model_info_file_name}\n"
        temp_string+=f"Bigstageresource_file_name:{self.Bigstageresource_file_name}\n"
        temp_string+=f"Ministagetime_file_name:{self.Ministagetime_file_name}\n"
        
        return temp_string+self.sum_string+"\n\n\n"
        



    
    
def experiment_one_group_parameters(args, file_sum, file_trace, print_level):

    weave_master=WeaveMaster(args, file_trace, print_level)
    weave_master.run()
    weave_master.print_job_time_info()
    if print_level>=1:
        
        print(f"The simulation end (system:{args.system}, strategy:{args.strategy}, file_sum:{file_sum != None}, file_trace:{file_trace != None}) !")

    if file_sum != None:
        out_string=weave_master.get_sum_info()
        file_sum.write(out_string)
        file_sum.flush()
        
    result_dict["sum_info"]= weave_master.get_sum_info()



def run_system(args):
    global result_dict
    
    #control parameters
    version=f"sim_v2.2.0_os{args.overshared_factor}"

    system=args.system
    strategy=args.strategy
    write_sum =False#args.write_sum
    write_trace = False#args.write_trace
    print_level=args.print_level

    cur_dir=os.path.dirname(os.path.abspath(__file__))
    parent_dir= os.path.dirname(os.path.abspath(cur_dir))
    # 格式化输出
    now_time= datetime.datetime.now()
    formatted_time = now_time.strftime('%m_%d_%H_%M_%S')
    sim_sum_file_name="Sim_Sum-"+system+"_"+strategy+"-"+version+"_"+formatted_time+".txt"
    sim_trace_file_name="Sim_Trace_"+system+"_"+strategy+"_"+version+"_"+formatted_time+".csv"
    if write_sum:
        file_sum=open(parent_dir+"/output/"+sim_sum_file_name,"w")
    else:
        file_sum=None

    if write_trace:
        file_trace = open(parent_dir + "/output/" + sim_trace_file_name, "w")
    else:
        file_trace=None

    experiment_one_group_parameters(args, file_sum, file_trace, print_level)

    if write_sum:
        file_sum.close()
    if write_trace:
        file_trace.close()
        
    return result_dict


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='simulation for DL training job')
    parser.add_argument("--system",default="Normal",type=str)
    parser.add_argument("--strategy", default="SRSF", type=str)
    parser.add_argument("--mps_flage", default="True", type=str)
    parser.add_argument("--sync_flage", default="True", type=str)
    parser.add_argument("--job_together_flage", default="True", type=str)
    parser.add_argument("--print_level", default=11, type=int)
    parser.add_argument("--node_kind", default="4*3090", type=str, help="cluster, 4*3090, 3*2080ti, 4*2080")
    parser.add_argument("--model_kind", default="all_model", type=str, help="cv_model, all_model")
    parser.add_argument("--gpu_mem_percent", default=0.9, type=float, help="because of GPU fragement")
    parser.add_argument("--node_num", default=1000, type=int, help="only for node kind is cluster")
    parser.add_argument("--job_num", default=100, type=int)
    parser.add_argument("--validation", default="True", type=str)
    parser.add_argument("--overshared_factor", default=3.0, type=float)
    parser.add_argument("--write_sum", action='store_true')
    parser.add_argument("--write_trace", action='store_true')
    parser.add_argument("--couple_init_iter_percent", default=0.2,type=float)
    parser.add_argument("--bucket_length", default=100000, type=int)
    args=parser.parse_args()


    #control parameters
    version=f"sim_v2.2.0_os{args.overshared_factor}"

    system=args.system
    strategy=args.strategy
    write_sum = True#args.write_sum
    write_trace = True#args.write_trace
    print_level=args.print_level

    cur_dir=os.path.dirname(os.path.abspath(__file__))
    parent_dir= os.path.dirname(os.path.abspath(cur_dir))
    # 格式化输出
    now_time= datetime.datetime.now()
    formatted_time = now_time.strftime('%m_%d_%H_%M_%S')
    sim_sum_file_name="Sim_Sum-"+system+"_"+strategy+"-"+version+"_"+formatted_time+".txt"
    sim_trace_file_name="Sim_Trace_"+system+"_"+strategy+"_"+version+"_"+formatted_time+".csv"
    if write_sum:
        file_sum=open(parent_dir+"/output/"+sim_sum_file_name,"w")
    else:
        file_sum=None

    if write_trace:
        file_trace = open(parent_dir + "/output/" + sim_trace_file_name, "w")
    else:
        file_trace=None

    experiment_one_group_parameters(args, file_sum, file_trace, print_level)

    if write_sum:
        file_sum.close()
    if write_trace:
        file_trace.close()
    
    