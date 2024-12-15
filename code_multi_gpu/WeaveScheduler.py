import math
import random
import os
import threading
from util import *


def get_strategy():
    strategy_all={}
    ####################第一个任务信息
    task_first={}
    task_first["MASTER_ADDR"]="10.26.128.51"
    task_first["MASTER_PORT"]=12355
    task_first["nnodes"]=1
    task_first["nprocs_per_node"]=2
    
    task_first["model_name"]="AlexNet"
    task_first["total_epochs"]=3
    task_first["batch_size"]=16
    task_first["worker_num"]=4
    
    task_first["max_sync_num"]=2
    
    
    #master特定信息
    task_first["master_spec"]={}
    task_first["master_spec"]["net_card"]="eno1"
    task_first["master_spec"]["node_rank"]=0
    task_first["master_spec"]["gpu_id_list"]=[1,2]
    #worker特定信息
    task_first["worker_spec"]={}
    task_first["worker_spec"]["net_card"]="eno1"
    task_first["worker_spec"]["node_rank"]=1
    task_first["worker_spec"]["gpu_id_list"]=[2,3]
    
    ####################第二个任务信息
    task_second={}
    task_second["MASTER_ADDR"]="10.26.128.51"
    task_second["MASTER_PORT"]=12345
    task_second["nnodes"]=1
    task_second["nprocs_per_node"]=2
    
    task_second["model_name"]="ResNet18"
    task_second["total_epochs"]=3
    task_second["batch_size"]=16
    task_second["worker_num"]=4
    
    task_second["max_sync_num"]=2
    #master特定信息
    task_second["master_spec"]={}
    task_second["master_spec"]["net_card"]="eno1"
    task_second["master_spec"]["node_rank"]=0
    task_second["master_spec"]["gpu_id_list"]=[1,2]
    #worker特定信息
    task_second["worker_spec"]={}
    task_second["worker_spec"]["net_card"]="eno1"
    task_second["worker_spec"]["node_rank"]=1
    task_second["worker_spec"]["gpu_id_list"]=[2,3]
    
    ##############共同共享内存名
    task_first["shm_name_list"]=[]
    task_second["shm_name_list"]=[]
    for i in range(task_first["max_sync_num"]):
        shm_name=generate_shm_name(16)
        task_first["shm_name_list"].append(shm_name)
        task_second["shm_name_list"].append(shm_name)
        
    strategy_all["task_first"]=task_first
    strategy_all["task_second"]=task_second
    return strategy_all


class WeaveSchedulor:
    def __init__(self,master, strategy,print_level=0):
        
        self.master=master
        self.strategy=strategy
        self.gpu_list=[i for i in range(7)]
        self.print_level=print_level
        
    
    def do_schedule(self,job_list):
        if self.print_level>2:
            print(f"start schedule ({self.strategy})...")
        if self.strategy=="random":
            rest_job=self.schedule_random(job_list)
        elif self.strategy=="over_sharing":
            rest_job=self.schedule_weave_over_sharing(job_list)
        else:
            print("strategy wrong!")
            exit(-1)
        return rest_job
    
    def schedule_random(self, job_list):
        
        for job in job_list:
            gpu_num=math.ceil(job.plan_gpu)
            select_gpu=random.sample(self.gpu_list, gpu_num)
            self.execute_schedule(job,select_gpu)
        return []
    
    def schedule_weave_over_sharing(self,job_list):
        multi_gpu_jobs, single_gpu_jobs=self.__over_sharing_get_multi_gpu_jobs(job_list)
        matched_jobs_list=self.__over_sharing_match_multi_gpu_jobs(multi_gpu_jobs)
        self.__over_sharing_select_gpu_for_multi_gpu_jobs(matched_jobs_list)
        
        
        # for job in job_list:
            
            
            
        return []
    
    
    
    
    def __over_sharing_get_multi_gpu_jobs(self,job_list):
        multi_gpu_jobs=[]
        single_gpu_jobs=[]
        for job in job_list:
            if job.is_multi_gpu():
                multi_gpu_jobs.append(job)
            else:
                single_gpu_jobs.append(job)
        return multi_gpu_jobs, single_gpu_jobs
    
    def __over_sharing_match_multi_gpu_jobs(self, multi_gpu_jobs):
        matched_job_name=set()
        complete_match_list=[]
        out_matched_jobs_list=[]
        for i_index in range(len(multi_gpu_jobs)):
            for j_index in range(i_index+1, len(multi_gpu_jobs)):
                job1=multi_gpu_jobs[i_index]
                job2=multi_gpu_jobs[j_index]
                epoch1=job1.total_epochs
                parrallel1=job1.parrallel_num
                [cpu11, mem11, gpu11, gmem11,time11]=self.master.analyze_2080_loader.get_job_values(job1,"stage_sample")
                [cpu12, mem12, gpu12, gmem12,time12]=self.master.analyze_2080_loader.get_job_values(job1,"stage_train")

                epoch2=job2.total_epochs
                parrallel2=job2.parrallel_num
                [cpu21, mem21, gpu21, gmem21,time21]=self.master.analyze_2080_loader.get_job_values(job2,"stage_sample")
                [cpu22, mem22, gpu22, gmem22,time22]=self.master.analyze_2080_loader.get_job_values(job2,"stage_train")
                
                epoch_factor=self.__over_sharing_cal_similarity(epoch1,epoch2)
                parrallel_factor=self.__over_sharing_cal_similarity(parrallel1,parrallel2)
                cpu_factor=self.__over_sharing_cal_similarity(cpu11+cpu22,cpu12+cpu21)
                mem_factor=self.__over_sharing_cal_similarity(mem11+mem22,mem12+mem21)
                gpu_factor=self.__over_sharing_cal_similarity(gpu11+gpu22,gpu12+gpu21)
                gmem_factor=self.__over_sharing_cal_similarity(gmem11+gmem22,gmem12+gmem21)
                time_factor=self.__over_sharing_cal_similarity(time11+time22,time12+time21)
                
                pack_resource=[max(cpu11+cpu22,cpu12+cpu21 ), max(mem11+mem22, mem12+mem21), max(gpu11+gpu22, gpu12+gpu21), max(gmem11+gmem22, gmem12+gmem21) ]
                
                simimlarity=epoch_factor+parrallel_factor+cpu_factor+mem_factor+gpu_factor+gmem_factor+time_factor
                
                complete_match_list.append([simimlarity,job1,job2,pack_resource])
        #按照匹配值高低进行提取
        complete_match_list.sort(key=lambda x:x[0], reverse=True)
        for i in range(len(complete_match_list)):
            job1=complete_match_list[i][1]
            job2=complete_match_list[i][2]
            
            if job1.job_name not in matched_job_name and job2.job_name not in matched_job_name:
                pack_resource=complete_match_list[i][3]
                matched_job_name.add(job1.job_name)
                matched_job_name.add(job2.job_name)
                
                out_matched_jobs_list.append([job1,job2,pack_resource])
                continue
                
            if i == len(complete_match_list):
                if job1.job_name not in matched_job_name:
                    matched_job_name.add(job1.job_name)
                    [cpu11, mem11, gpu11, gmem11,time11]=self.master.analyze_2080_loader.get_job_values(job1,"stage_sample")
                    [cpu12, mem12, gpu12, gmem12,time12]=self.master.analyze_2080_loader.get_job_values(job1,"stage_train")

                    out_matched_jobs_list.append([job1, [max(cpu11,cpu12), max(mem11,mem12), max(gpu11,gpu12), max(gmem11,gmem12)]])
                if job2.job_name not in matched_job_name:
                    matched_job_name.add(job2.job_name)
                    [cpu21, mem21, gpu21, gmem21,time21]=self.master.analyze_2080_loader.get_job_values(job2,"stage_sample")
                    [cpu22, mem22, gpu22, gmem22,time22]=self.master.analyze_2080_loader.get_job_values(job2,"stage_train")

                    out_matched_jobs_list.append([job2, [max(cpu21,cpu22), max(mem21,mem22), max(gpu21,gpu22), max(gmem21,gmem22)]])
        #判断输出是否包含所有jobs
        if len(matched_job_name) != multi_gpu_jobs:
            print("__over_sharing_match_multi_gpu_jobs wrong!")
            exit(256)
        return out_matched_jobs_list
                
                
    def __over_sharing_cal_similarity(self, var1, var2):
        return 1-(abs(var1-var2)/max(var1,var2))
        
    
    def __over_sharing_select_gpu_for_multi_gpu_jobs(self, matched_jobs_list):
        #matched_jobs_list = [job1, job2, [cpu, mem, gpu, gmem] ] or [job1, [cpu, mem, gpu, gmem] ] 
        for matched_jobs in matched_jobs_list:
            if len(matched_jobs) == 3:
                job1=matched_jobs[0]
                job2=matched_jobs[1]
                pack_resource=matched_jobs[2]
            else:
                job1=matched_jobs[0]
                job2=None
                pack_resource=matched_jobs[1]
            
            
            
    
    
    
    def execute_schedule(self,job, select_gpu_list):
        world_size=len(select_gpu_list)
        master_gpu_id_list=[x for x in select_gpu_list if x <4]
        worker_gpu_id_list=[x-4 for x in select_gpu_list if x >= 4]
        
        nprocs_list=[len(master_gpu_id_list), len(worker_gpu_id_list)]
        gpu_id_list=[master_gpu_id_list,worker_gpu_id_list]

        is_cross=True if len(master_gpu_id_list)>0 and len(worker_gpu_id_list)>0 else False

        if len(master_gpu_id_list)>0:
            self.job_set_execute_info(job, True, is_cross, world_size, nprocs_list, gpu_id_list )
            self.master.execute_job_in_master(job)
            
        if len(worker_gpu_id_list)>0:
            self.job_set_execute_info(job, False, is_cross, world_size, nprocs_list, gpu_id_list )
            self.master.send_job_to_worker(job)
            
    def job_set_execute_info(self, job, is_master, is_cross, world_size, nprocs_list, gpu_id_list):
        if is_master:
            node_rank=0
            net_card="eno2"
        else:
            node_rank=1
            net_card="eno1"
        

        if is_cross and is_master:
            while is_port_in_use(self.master.master_port):
                if self.print_level>9:
                    print("change port")
                self.master.master_port+=1


        if is_cross or is_master:
            MASTER_ADDR="10.26.128.115"
            if is_master:
                MASTER_PORT=self.master.master_port
                self.master.master_port+=1
            else:
                MASTER_PORT=self.master.master_port-1
            
        elif (not is_cross) and (not is_master):
            MASTER_ADDR="10.26.128.51"
            MASTER_PORT=self.master.worker_port
            self.master.worker_port+=1
        
            
            
        job.set_execute_info(MASTER_ADDR, MASTER_PORT, net_card, node_rank,world_size ,nprocs_list, gpu_id_list)
        
    
    
    
    
    