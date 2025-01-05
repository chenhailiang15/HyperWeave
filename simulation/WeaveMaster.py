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

#cpu, gpu 按照百分比表示需求和剩余，即1个GPU 表示为100
#mem, gmem按照存储单位表示，本平台中使用MB

####################已有的问题#####################
#Weave 和 Muri的时间分析结果不一样



#################################################
class WeaveMaster:
    
    def __init__(self,system, strategy, mps_flage, sync_flage, print_level=0):
        
        
        ##################################################《--设置区域--》开始####################################################

        self.system=system           #"Muri" or "Normal"
        self.schedule_strategy=strategy   # "FIFO", "SRTF"，"SRSF", "BNPF"   Bucket-based non-blocking parallel first
        self.weave_sync_mode=sync_flage
        self.MPS_mode=mps_flage              #需要最好手动确认
        
        self.overshared_factor=1   #等于1存在GPU资源不够的情况
        self.clock_time_factor = 1
        self.job_time_factor = 1
        self.job_ddl_factor = 1  # ddl是任务持续时间的job_ddl_factor倍
        self.schedule_interval = 3600

        self.print_level=print_level

        self.model_name_list=model_list_g    #
        self.batch_size_dict=model_to_batch_size_g

        self.max_cross=1           #最大跨node任务数量
        self.max_gpu_cross=1       #最大跨GPU任务数量（单node）

        self.ali_trace_node_info_file_name="sim_ali_trace_machine_info.csv"
        self.ali_trace_job_info_file_name="sim_ali_trace_job_info.csv"
        self.analyze_file_name="Analyzer-NVIDIA_GeForce_RTX_2080-tim_12_19_16_38_14.csv"
        self.model_time_file_name="Muri_Analyzer-NVIDIA_GeForce_RTX_2080_Ti-tim_12_26_15_24_41.csv"
        self.model_info_file_name="Full_model_info_12_27_21_27_41.txt"
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
        #################################

        if self.print_level>0:
            print("load ali trace (node info)...")
        self.init_node(self.ali_trace_node_info_file_name)

        if self.print_level>0:
            print("load ali trace (job info)...")
        self.load_ali_trace(self.ali_trace_job_info_file_name)
        
        if self.print_level>0:
            print("init analyze loader...")
        self.analyze_loader=AnalyzeDataLoader(self.analyze_file_name, print_level)
        self.analyze_loader.load_time_csv(self.model_time_file_name)


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
        node_info_pd = self.load_csv(file_name,header=0)
        self.node_num = len(node_info_pd)
        for index in range(len(node_info_pd)):
            name=node_info_pd.loc[index, "machine"]
            cpu_cap= node_info_pd.loc[index, "cap_cpu"]*100
            mem_cap=node_info_pd.loc[index, "cap_mem"]*1024
            gpu_cap=node_info_pd.loc[index, "cap_gpu"]
            if gpu_cap==0:
                gmem_cap=0
            else:
                gmem_cap=gpu_mem_dict[node_info_pd.loc[index, "gpu_type"]]*1024
            node=Node(self, self.env,name,index, self.overshared_factor, self.print_level)
            node.set_init_resouce( cpu_cap, mem_cap, gpu_cap, gmem_cap )
            self.nodes.append(node)

            
    def init_model_info(self):
        file=open(get_dataset_dir()+"exp_data/"+self.model_info_file_name,"r")
        for line in file.readlines():
            self.model_info_list.append(line[0:-1])
            self.model_info_list_max+=1
        
        
        
        
    def load_ali_trace(self,file_name):
        ali_trace_pd=self.load_csv(file_name,header=0)
        min_start_time=ali_trace_pd["start_time_j"].min()
        ali_trace_pd["start_time"]=(ali_trace_pd["start_time_j"]-min_start_time)/self.clock_time_factor
        self.ali_trace_pd=ali_trace_pd.sort_values(by="start_time")

    def load_csv(self, file_name,header=None):
        dataset_dir=get_dataset_dir()
        data_pd=pd.read_csv(dataset_dir+"exp_data"+"/"+file_name,header=header)
        return data_pd


    def run(self):
        self.start_time_real_world=time.time()
        self.env.process(self.job_come())
        self.env.process(self.schedule())
        self.env.process(self.print_system_state())
        self.env.run(until=99999999)

    # job到来的函数，持续运行，直到读取的文件中的job结束
    def job_come(self):

        self.start_time = self.env.now
        self.wait_schedule_queue = queue.Queue()
        self.job_come_flage = True
        for index in range(len(self.ali_trace_pd)):
            job = self.generate_job(self.ali_trace_pd.iloc[index, :], self.job_come_num)
            if job == None:  # 由于数据原因，可能无法生成Job，因此跳过
                continue

            if self.print_level >= 2:
                print(f"time: {self.env.now}\tjob {job.job_idx} \tcome ( detailed info :{job.job_key_info()})")
            self.wait_schedule_queue.put(job)
            self.job_come_num += 1
            # if index>=100:
            #     break

            if index + 1 < len(self.ali_trace_pd):
                yield self.env.timeout(self.ali_trace_pd.loc[index + 1, "start_time"] - self.ali_trace_pd.loc[index, "start_time"])

        self.job_come_flage = False

        return

    def generate_job(self, ali_trace,job_idx):
        
        if ali_trace["cpu_usage"]==0 or ali_trace["avg_mem"]==0:
            return None

        # while True: #从自己生成的模型信息中获取一个，满足持续时间要求
        model_name=self.model_info_list[self.model_info_list_index%self.model_info_list_max].split("-")[0]
        batch_size=int(self.model_info_list[self.model_info_list_index%self.model_info_list_max].split("-")[1])
        self.model_info_list_index+=1

        plan_gpu=ali_trace["plan_gpu"]

        parrallel_num=math.ceil(min(plan_gpu, 400)/100)
        model_info=model_name+"-"+str(batch_size)+"-"+str(parrallel_num)
        duration_time=ali_trace["duration_s"]/self.job_time_factor
        init_time=self.analyze_loader.get_value(model_info,"stage_init","time")
        epoch_time=self.analyze_loader.get_value(model_info,"stage_sample","time")+self.analyze_loader.get_value(model_info,"stage_train","time")
        model_duration_time=init_time+epoch_time

        if model_duration_time<duration_time:
            print("duration time is too small...")
            return None

        
        
        total_epochs=math.ceil((ali_trace["duration_s"]/self.job_time_factor-init_time)/epoch_time)
        each_batch_time=self.analyze_loader.get_time_value(model_info, 1)+self.analyze_loader.get_time_value(model_info, 2)+self.analyze_loader.get_time_value(model_info, 3)
        batch_num=math.ceil(self.analyze_loader.get_value(model_info,"stage_train","time")/each_batch_time)




        job = Job(job_idx, self.system)
        job_name = ali_trace["job_name"]
        job.set_model_info(job_name, model_name,total_epochs, batch_size)
        job.batch_num=batch_num    #设置batch numbere
        job.instance_num=int(ali_trace["inst_num"])

        #设置各阶段时间消耗
        job.time_init=self.analyze_loader.get_time_value(model_info,0)
        job.time_init_iter=self.analyze_loader.get_value(model_info,"stage_sample","time")
        job.time_get_data=self.analyze_loader.get_time_value(model_info,1)
        job.time_forward_back=self.analyze_loader.get_time_value(model_info,2)
        job.time_commu=self.analyze_loader.get_time_value(model_info,3)
        #设置计划资源使用量
        job.set_plan_resource(ali_trace["plan_cpu"], ali_trace["plan_mem"], ali_trace["plan_gpu"])
        #设置各阶段实际资源使用量[init stage, pre-iteration stage, iteration stage]
        job.used_resource_cpu = [ali_trace["cpu_usage"]*self.analyze_loader.get_value(model_info,"stage_init", "cpu")/self.analyze_loader.get_value(model_info,"stage_train", "cpu"),\
                                 ali_trace["cpu_usage"]*self.analyze_loader.get_value(model_info,"stage_sample", "cpu")/self.analyze_loader.get_value(model_info,"stage_train", "cpu"),\
                                 ali_trace["cpu_usage"]]
        job.used_resource_mem = [1024*ali_trace["avg_mem"]*self.analyze_loader.get_value(model_info,"stage_init", "mem")/self.analyze_loader.get_value(model_info,"stage_train", "mem"),\
                                 1024*ali_trace["avg_mem"]*self.analyze_loader.get_value(model_info,"stage_sample", "mem")/self.analyze_loader.get_value(model_info,"stage_train", "mem"),\
                                 1024*ali_trace["avg_mem"]]
        job.used_resource_gpu = [ali_trace["gpu_wrk_util"]*self.analyze_loader.get_value(model_info,"stage_init", "gpu")/self.analyze_loader.get_value(model_info,"stage_train", "gpu"),\
                                 ali_trace["gpu_wrk_util"]*self.analyze_loader.get_value(model_info,"stage_sample", "gpu")/self.analyze_loader.get_value(model_info,"stage_train", "gpu"),\
                                 ali_trace["gpu_wrk_util"]]
        job.used_resource_gmem = [1024*ali_trace["avg_gpu_wrk_mem"]*self.analyze_loader.get_value(model_info,"stage_init", "gmem")/self.analyze_loader.get_value(model_info,"stage_train", "gmem"),\
                                 1024*ali_trace["avg_gpu_wrk_mem"]*self.analyze_loader.get_value(model_info,"stage_sample", "gmem")/self.analyze_loader.get_value(model_info,"stage_train", "gmem"),\
                                 1024*ali_trace["avg_gpu_wrk_mem"]]

        #设置job到达时间
        arrive_time = self.env.now
        ddl_time = arrive_time + duration_time * self.job_ddl_factor
        job.set_arrive_time(arrive_time)
        job.set_ddl_time(ddl_time)
        job.set_duration_time(duration_time)
        for index in range(int(ali_trace["inst_num"])):
            instance=Instance(job, index, self.system)
            job.instance_list.append(instance)
        return job
        
    #调度子线程，间隔schedule_interval（秒）后，执行一次调度。未调度成功的job需要返回，重新放入队列
    #调度停止的条件是job不再到来（self.job_come_flage=False），并且队列为空(qsize==0)
    def schedule(self):
        yield self.env.timeout(self.schedule_interval)
        if self.job_come_flage or self.wait_schedule_queue.qsize()>0:
            wait_schedule_list=[]
            # self.queue_length.append(self.wait_schedule_queue.qsize())

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

            self.nodes[instance_t.node_rank].execute_instance(instance_t)

    
    def statistic_end_instance(self,instance):
        if self.print_level>3:
            print("end a instance:", instance.instance_name)

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

                if instance.job.succeed_instance_num+instance.job.failed_instance_num== instance.job.instance_num:
                    self.job_end_num+=1
                    self.job_dealing_num-=1
                    if instance.job.succeed_instance_num == instance.job.instance_num:
                        self.succeed_job_num+=1
                        self.job_wait_time_list.append(instance.job.start_time-instance.job.arrive_time)
                        self.job_complete_time_list.append(self.env.now-instance.job.arrive_time)
                    else:
                        self.failed_job_num+=1
                    
            if self.job_come_flage==False and self.job_come_num == self.job_end_num and self.command_start_num == self.command_end_num:
                print("event set")
                self.end_event.set()
                self.set_makespan()

            # self.print_current_state()

    def print_system_state(self):
        self.print_current_state()
        yield self.env.timeout(10000)
        if not self.end_event.is_set():
            self.env.process(self.print_system_state())

    def print_current_state(self):
        self.update_job_not_start_num()
        # if self.job_come_num!=self.job_not_start_num+self.job_start_num:
        #     a=1
        assert self.job_come_num==self.job_not_start_num+self.job_start_num
        out_string=f"************************current status (now:{self.env.now}) *******************************\n"
        out_string+=f"job come number:{self.job_come_num}\tjob not start number:{self.job_not_start_num}\n"
        out_string+=f"job start number:{self.job_start_num}\tjob dealing number:{self.job_dealing_num}\tjob end number:{self.job_end_num}(S:{self.succeed_job_num}/F:{self.failed_job_num})\n"
        out_string+=f"instance start number:{self.instance_start_num}\tinstance dealing number:{self.instance_dealing_num}\tinstance end number:{self.instance_end_num}\n"
        out_string+=f"command start number:{self.command_start_num}\tcommand dealing number:{self.command_dealing_num}\tcommand end number:{self.command_end_num}\n\n"
        print(out_string)

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




    def set_makespan(self):
        self.makespan=self.env.now-self.start_time

            
    def get_sum_info(self):
        
        
        temp_string=f"****************************************************{self.schedule_strategy}***********************************************************\n"
        temp_string+="    ^^^^    ^^^^    ^^^^    ^^^^    ^^^^    ^^^^    parameters in experiment    ^^^^    ^^^^    ^^^^    ^^^^    ^^^^    ^^^^    \n"
        temp_string+=f"system name:{self.system}, \tMPS:{self.MPS_mode}, \tSync:{self.weave_sync_mode}\n"
        temp_string+=f"overshared_factor:{self.overshared_factor}, \tmax_cross:{self.max_cross}, \tmax_gpu_cross:{self.max_gpu_cross}\n"
        temp_string+=f"clock_time_factor:{self.clock_time_factor}, \tjob_time_factor:{self.job_time_factor}, \tjob_ddl_factor:{self.job_ddl_factor}\n"
        temp_string+=f"model_name_list:{self.model_name_list}\n"
        temp_string+=f"batch_size_dict:{self.batch_size_dict}\n"
        temp_string+=f"schedule_interval:{self.schedule_interval}\n"
        temp_string+=f"ali_trace_job_info_file_name:{self.ali_trace_job_info_file_name}\n"
        temp_string+=f"ali_trace_node_info_file_name:{self.ali_trace_node_info_file_name}\n"
        temp_string+=f"analyze_file_name:{self.analyze_file_name}\n"
        temp_string+="    ^^^^    ^^^^    ^^^^    ^^^^    ^^^^    time info in following    ^^^^    ^^^^    ^^^^    ^^^^    ^^^^    \n"
        temp_string+=f"time cost(s){time.time()-self.start_time_real_world}\n"
        temp_string+=f"all job num:{self.job_come_num}, \tsucceed job num:{self.succeed_job_num}, \tfailed job num:{self.failed_job_num}\n"
        temp_string+=f"makespan:{self.makespan}\n"
        temp_string+=f"queue length{self.queue_length}\n"
        
        return temp_string+self.sum_string+"\n\n\n"
        



    
    
def experiment_one_group_parameters(system, strategy, mps_flage, sync_flage, file, print_level):

    weave_master=WeaveMaster(system, strategy, mps_flage, sync_flage, print_level)
    weave_master.run()
    weave_master.print_job_time_info()

    if file != None:
        out_string=weave_master.get_sum_info()
        file.write(out_string)
        file.flush()

    print(f"The simulation end (system:{strategy}, strategy:{strategy}, mps:{mps_flage}, sync:{sync_flage}, file:{file != None}) !")




if __name__=="__main__":
    # parser = argparse.ArgumentParser(description='simulation for DL training job')
    # parser.add_argument("--system",default="normal",type=str)
    # parser.add_argument("--strategy", default="FIFO", type=str)
    #
    # parser.add_argument("--write_flage", action='store_true')
    # parser.add_argument("--mps_flage", action='store_true')
    # parser.add_argument("--sync_flage", action='store_true')


    #control parameters
    version="sim_v1.0.0"
    write_flage=True
    system="Normal"
    strategy="FIFO"
    mps_flage=False
    sync_flage=False
    print_level=0

    parent_dir= os.path.dirname(os.path.abspath(os.curdir))
    # 格式化输出
    now_time= datetime.datetime.now()
    formatted_time = now_time.strftime('%m_%d_%H_%M_%S')
    sum_info_file_name="SumInfo_Weave_"+version+"_"+formatted_time+".txt"
    if write_flage:
        file=open(parent_dir+"/output/"+sum_info_file_name,"w")
    else:
        file=None

    experiment_one_group_parameters(system, strategy, mps_flage, sync_flage, file,print_level)
    # experiment_all(file, "SRTF",formatted_time, version)
    # experiment_all(file, "SRSF",formatted_time, version)
    if write_flage:
        file.close()
    
    