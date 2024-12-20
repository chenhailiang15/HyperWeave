import math
import random
import os
import threading
from util import *
import copy
import time

class WeaveSchedulor:
    def __init__(self,master, print_level=0):
        
        self.master=master
        self.strategy=master.schedule_strategy
        self.gpu_list=[i for i in range(7)]
        self.print_level=print_level
        
    
    def do_schedule(self,job_list):
        if self.print_level>2:
            print(f"start schedule ({self.strategy})...")
            
        if self.master.start_weave:
            rest_job=self.schedule_weave_over_sharing(job_list)
        else:
            if self.strategy=="FIFO":
                rest_job=self.schedule_FIFO(job_list)
            elif self.strategy=="SRTF":
                rest_job=self.schedule_SRTF(job_list)
            elif self.strategy=="SRSF":
                rest_job=self.schedule_SRSF(job_list)
            else:
                print("strategy wrong!")
                exit(-1)
        return rest_job
    
    
    
    def schedule_FIFO(self, job_list):
        #按照到来的先后顺序排序
        job_list.sort(key=lambda x: x.arrive_time)
        rest_job=self.schedule_ordered_job_list(job_list)
            
        return rest_job
    
    def schedule_SRTF(self, job_list):
        time_now=time.time()
        #按照到来的先后顺序排序
        job_list.sort(key=lambda x: x.ddl_time-time_now-x.duration_time)
        rest_job=self.schedule_ordered_job_list(job_list)
            
        return rest_job
    
    def schedule_SRSF(self, job_list):
        
        time_now=time.time()
        # for job in job_list:
        #     job.set_schedule_order(time_now)
            
        #按照到来的先后顺序排序
        job_list.sort(key=lambda x: (x.ddl_time-time_now-x.duration_time)*x.parallel_num)
        rest_job=self.schedule_ordered_job_list(job_list)
        return rest_job
    
    
    def schedule_ordered_single_job_list(self, job_list):
        #初始化未被调度的job
        rest_job=[]
        #是否继续调度的标志，当遇到一个无法调度的任务时，停止调度等待下一轮调度，将剩余的job返回
        continue_schedule_flage=True
        #循环调度
        for job in job_list:
            if continue_schedule_flage:
                #正常调度
                plan_resource=[job.plan_cpu, job.plan_mem/job.parallel_num, job.plan_gpu/job.parallel_num, 0]
                satisfy_gpu_list=self.master.monitor.get_satisfy_gpu(plan_resource)
                all_satisfy_gpu_num=0
                for [node_index, score, temp_gpu_list] in satisfy_gpu_list:
                    all_satisfy_gpu_num+=len(temp_gpu_list)
                if all_satisfy_gpu_num>=job.parallel_num:
                    selected_gpu_id_list,_,_= self.__over_sharing_select_gpu(job.parallel_num, 0, satisfy_gpu_list)
                    self.master.monitor.alloc_resource(job, None, selected_gpu_id_list, plan=True)
                    self.execute_schedule(job, selected_gpu_id_list, False, {})
                else:
                    rest_job.append(job)
                    continue_schedule_flage=False
            else:
                rest_job.append(job)
        return rest_job
    
    
    #over share 调度主线
    def schedule_weave_over_sharing(self,job_list):
        multi_gpu_jobs, single_gpu_jobs=self.__over_sharing_classify_jobs(job_list)         #job 根据其并行数量（GPU数量）分类为多GPU任务和单GPU任务
        matched_multi_jobs_list=self.__over_sharing_match_jobs(multi_gpu_jobs)                    #对多GPU任务进行匹配
        matched_single_jobs_list=self.__over_sharing_match_jobs(single_gpu_jobs)                   #对单GPU任务进行匹配
        
        if self.strategy=="FIFO":
            rest_job=self.schedule_weave_FIFO(matched_multi_jobs_list, matched_single_jobs_list)
        elif self.strategy=="SRTF":
            rest_job=self.schedule_weave_SRTF(matched_multi_jobs_list, matched_single_jobs_list)
        elif self.strategy=="SRSF":
            rest_job=self.schedule_weave_SRSF(matched_multi_jobs_list, matched_single_jobs_list)
        else:
            print("strategy wrong!")
            exit(-1)

        return rest_job
    
    
    def schedule_weave_FIFO(self, multi_gpu_jobs, single_gpu_jobs):
        order_matched_jobs=[]
        matched_jobs= multi_gpu_jobs+single_gpu_jobs
        for [job1,job2,pack_resource] in matched_jobs:
            if job2 != None:
                arrive_time=min(job1.arrive_time, job2.arrive_time)
            else:
                arrive_time=job1.arrive_time
            order_matched_jobs.append([job1, job2, pack_resource, arrive_time])
        #matched_jobs排序
        order_matched_jobs.sort(key=lambda x : x[3])
        rest_job=self.schedule_weave_ordered_matched_job_list(order_matched_jobs)
        return rest_job
    
    
    
    
    
    
    
    def schedule_weave_SRTF(self, multi_gpu_jobs, single_gpu_jobs):
        order_matched_jobs=[]
        time_now=time.time()
        matched_jobs= multi_gpu_jobs+single_gpu_jobs
        for [job1,job2,pack_resource] in matched_jobs:
            if job2 != None:
                rest_time1=job1.ddl_time-time_now-job1.duration_time
                rest_time2=job2.ddl_time-time_now-job2.duration_time
                rest_time=min(rest_time1, rest_time2)
            else:
                rest_time=job1.ddl_time-time_now-job1.duration_time  
            order_matched_jobs.append([job1, job2, pack_resource, rest_time])
        
        #match_jobs排序
        order_matched_jobs.sort(key=lambda x : x[3])
        rest_job=self.schedule_weave_ordered_matched_job_list(order_matched_jobs)
        return rest_job
        
    def schedule_weave_SRSF(self, multi_gpu_jobs, single_gpu_jobs):
        order_matched_jobs=[]
        time_now=time.time()
        matched_jobs= multi_gpu_jobs+single_gpu_jobs
        for [job1,job2,pack_resource] in matched_jobs:
            if job2 != None:
                rest_time1=(job1.ddl_time-time_now-job1.duration_time)*job1.parallel_num
                rest_time2=(job2.ddl_time-time_now-job2.duration_time)*job2.parallel_num
                rest_time=min(rest_time1, rest_time2)
            else:
                rest_time=job1.ddl_time-time_now-job1.duration_time  
            order_matched_jobs.append([job1, job2, pack_resource, rest_time])
        
        #match_jobs排序
        order_matched_jobs.sort(key=lambda x : x[3])
        rest_job=self.schedule_weave_ordered_matched_job_list(order_matched_jobs)
        return rest_job
        
    def schedule_weave_ordered_matched_job_list(self, match_jobs):
        #初始化未被调度的job
        rest_job=[]
        #是否继续调度的标志，当遇到一个无法调度的任务时，停止调度等待下一轮调度，将剩余的job返回
        continue_schedule_flage=True
        #循环调度
        for [job1,job2,pack_resource,order_value] in match_jobs:
            if continue_schedule_flage:
                #正常调度
                #[[node_index, score, [[gpu_id, score],...]],...]
                satisfy_gpu_list=self.master.monitor.get_satisfy_gpu(pack_resource)
                #对优先级进行排序
                satisfy_gpu_list_new=[]
                all_satisfy_gpu_num=0
                for [node_index, score, temp_gpu_list] in satisfy_gpu_list:
                    score=score+len(temp_gpu_list)*10-self.master.nodes[node_index].get_over_corss_num()*10
                    #调整优先级，score原始是GPU剩余量百分比的和
                    satisfy_gpu_list_new.append([node_index, score, temp_gpu_list])
                    all_satisfy_gpu_num+=len(temp_gpu_list)
                satisfy_gpu_list_new.sort(key=lambda x:x[1], reverse=True)
                
                
                if job2 != None :
                    max_parallel=max(job1.parallel_num, job2.parallel_num)
                    if all_satisfy_gpu_num<max_parallel:
                        rest_job.append(job1)
                        rest_job.append(job2)
                        continue_schedule_flage=False
                        continue
                    job1_rest_gpu=job1.parallel_num
                    job2_rest_gpu=job2.parallel_num
                    
                    job1_gpu_id_list,job2_gpu_id_list,shm_name_dict= self.__over_sharing_select_gpu(job1_rest_gpu, job2_rest_gpu, satisfy_gpu_list_new)
                    # scheduled_jobs.append([job1, job1_gpu_id_list, shm_name_dict])
                    # scheduled_jobs.append([job2, job2_gpu_id_list, shm_name_dict])
                    [cpu, mem, gpu, gmem] =pack_resource
                    job1.set_pack_resource(math.ceil(cpu), math.ceil(mem), math.ceil(gpu), math.ceil(gmem), job2.job_name)
                    job2.set_pack_resource(math.ceil(cpu), math.ceil(mem), math.ceil(gpu), math.ceil(gmem), job1.job_name)
                    
                    if job1.parallel_num>= job2.parallel_num:
                        max_couple_job_gpu_id_list=job1_gpu_id_list
                    else:
                        max_couple_job_gpu_id_list=job2_gpu_id_list
                    self.master.monitor.alloc_resource(job1, job2,max_couple_job_gpu_id_list)
                    if self.master.sync_mode==False:
                        shm_name_dict={}
                    self.execute_schedule(job1, job1_gpu_id_list, True, shm_name_dict)
                    self.execute_schedule(job2, job2_gpu_id_list, False,  shm_name_dict)
                else:
                    max_parallel=job1.parallel_num
                    if all_satisfy_gpu_num<max_parallel:
                        rest_job.append(job1)
                        continue_schedule_flage=False
                        continue
                    job1_rest_parallel_num=job1.parallel_num
                    
                    job1_gpu_id_list, _, _= self.__over_sharing_select_gpu(job1_rest_parallel_num, 0, satisfy_gpu_list_new)   
                    # scheduled_jobs.append([job1, job1_gpu_id_list, {}])
                    [cpu, mem, gpu, gmem] =pack_resource
                    job1.set_pack_resource(math.ceil(cpu), math.ceil(mem), math.ceil(gpu), math.ceil(gmem) )
                    
                    self.master.monitor.alloc_resource(job1, None,job1_gpu_id_list)
                    self.execute_schedule(job1, job1_gpu_id_list, False, {})
            
            else:
                if job2 !=None:
                    rest_job.append(job2)
                rest_job.append(job1)
                
        return rest_job
        
        
    #任务分类 单GPU和多GPU任务
    def __over_sharing_classify_jobs(self,job_list):     
        multi_gpu_jobs=[]
        single_gpu_jobs=[]
        for job in job_list:
            if job.is_multi_gpu():
                multi_gpu_jobs.append(job)
            else:
                single_gpu_jobs.append(job)
        return multi_gpu_jobs, single_gpu_jobs
    
    #任务匹配
    def __over_sharing_match_jobs(self, jobs_list):
        complete_match_list=[]
        out_matched_jobs_list=[]
        if len(jobs_list)==0:
            return out_matched_jobs_list
        if len(jobs_list) == 1:
            job1=jobs_list[0]
            pack_resource=self.master.analyze_loader.get_job_pack_resource(job1)
            out_matched_jobs_list.append([job1, None, pack_resource])
            return out_matched_jobs_list
        
        #计算两两的匹配值
        for i_index in range(len(jobs_list)):
            for j_index in range(i_index+1, len(jobs_list)):
                job1=jobs_list[i_index]
                job2=jobs_list[j_index]
                epoch1=job1.total_epochs
                parallel1=job1.parallel_num
                [cpu10, mem10, gpu10, gmem10,time10]=self.master.analyze_loader.get_job_values(job1,"stage_init")
                [cpu11, mem11, gpu11, gmem11,time11]=self.master.analyze_loader.get_job_values(job1,"stage_sample")
                [cpu12, mem12, gpu12, gmem12,time12]=self.master.analyze_loader.get_job_values(job1,"stage_train")

                epoch2=job2.total_epochs
                parallel2=job2.parallel_num
                [cpu20, mem20, gpu20, gmem20,time20]=self.master.analyze_loader.get_job_values(job2,"stage_init")
                [cpu21, mem21, gpu21, gmem21,time21]=self.master.analyze_loader.get_job_values(job2,"stage_sample")
                [cpu22, mem22, gpu22, gmem22,time22]=self.master.analyze_loader.get_job_values(job2,"stage_train")
                
                epoch_factor=self.__over_sharing_cal_similarity(epoch1,epoch2)
                parallel_factor=self.__over_sharing_cal_similarity(parallel1,parallel2)
                cpu_factor=self.__over_sharing_cal_similarity(cpu11+cpu22,cpu12+cpu21)
                mem_factor=self.__over_sharing_cal_similarity(mem11+mem22,mem12+mem21)
                gpu_factor=self.__over_sharing_cal_similarity(gpu11+gpu22,gpu12+gpu21)
                gmem_factor=self.__over_sharing_cal_similarity(gmem11+gmem22,gmem12+gmem21)
                time_factor=self.__over_sharing_cal_similarity(time11+time22,time12+time21)
                
                pack_resource=[max(cpu10+cpu20,cpu11+cpu22,cpu12+cpu21 ), max(mem10+mem20, mem11+mem22, mem12+mem21), max(gpu10+gpu20, gpu11+gpu22, gpu12+gpu21), max(gmem10+gmem20, gmem11+gmem22, gmem12+gmem21)]
                max_resource=self.master.monitor.get_max_resource()
                result=all(max_value>=need_value for max_value, need_value in zip(max_resource, pack_resource))
                if result:
                    similarity=epoch_factor+parallel_factor+cpu_factor+mem_factor+gpu_factor+gmem_factor+time_factor
                else:
                    similarity=0
                complete_match_list.append([similarity,job1,job2,pack_resource])          #存储匹配值，后续用于匹配
                
        #按照匹配值高低进行提取
        matched_job_name=set()
        single_job=None
        single_job_pack_resource=None
        complete_match_list.sort(key=lambda x:x[0], reverse=True)
        for [similarity,job1,job2,pack_resource] in complete_match_list:

            if job1.job_name not in matched_job_name and job2.job_name not in matched_job_name:
                if similarity==0:  #如果为0，表示当前匹配已经超出机器最大资源容量， 放弃匹配
                    single_job1_pack_resource=self.master.analyze_loader.get_job_pack_resource(job1)
                    out_matched_jobs_list.append([job1,None,single_job1_pack_resource])
                    single_job2_pack_resource=self.master.analyze_loader.get_job_pack_resource(job2)
                    out_matched_jobs_list.append([job2,None,single_job2_pack_resource])
                else:
                    out_matched_jobs_list.append([job1,job2,pack_resource])
                matched_job_name.add(job1.job_name)
                matched_job_name.add(job2.job_name)
                continue
            
            elif job1.job_name not in matched_job_name:
                single_job=job1
                single_job_pack_resource=self.master.analyze_loader.get_job_pack_resource(job1)
            elif job2.job_name not in matched_job_name:
                single_job=job2
                single_job_pack_resource=self.master.analyze_loader.get_job_pack_resource(job2)
                
            
        #判断输出是否包含所有jobs
        if len(matched_job_name) != len(jobs_list):
            if single_job.job_name not in matched_job_name:
                out_matched_jobs_list.append([single_job, None, single_job_pack_resource])
            else:
                print("__over_sharing_match_multi_gpu_jobs wrong!")
                exit(256)
        return out_matched_jobs_list
                
    #匹配值计算的子函数，匹配效果越好，值越接近1
    def __over_sharing_cal_similarity(self, var1, var2):
        return 1-(abs(var1-var2)/max(var1,var2))
                
                
    def __over_sharing_select_gpu(self, job1_rest_gpu, job2_rest_gpu, satisfy_gpu_list_new) :
        job1_gpu_id_list=[]
        job2_gpu_id_list=[]
        shm_name_dict={}
        for [node_index, score, temp_gpu_list] in satisfy_gpu_list_new:
            job1_temp_gpu_id_list=[]
            job2_temp_gpu_id_list=[]
            shm_name_dict_temp={}
            for [gpu_index, ave_per] in temp_gpu_list:
                if job1_rest_gpu>0 and job2_rest_gpu>0:
                    shm_name=generate_shm_name()
                    shm_name_dict_temp[gpu_index]=shm_name
                    job1_temp_gpu_id_list.append(gpu_index)
                    job2_temp_gpu_id_list.append(gpu_index)
                    job1_rest_gpu-=1
                    job2_rest_gpu-=1
                elif job1_rest_gpu>0 and job2_rest_gpu==0:
                    job1_temp_gpu_id_list.append(gpu_index)
                    job1_rest_gpu-=1
                elif job1_rest_gpu==0 and job2_rest_gpu>0:
                    job2_temp_gpu_id_list.append(gpu_index)
                    job2_rest_gpu-=1
                elif job1_rest_gpu==0 and job2_rest_gpu==0:
                    break
                else:
                    print("(monitor) wrong!")
                    exit(256)
            job1_gpu_id_list.append([node_index, job1_temp_gpu_id_list])
            job2_gpu_id_list.append([node_index, job2_temp_gpu_id_list])
            shm_name_dict[node_index]=shm_name_dict_temp
            if job1_rest_gpu==0 and job2_rest_gpu==0:
                break
        return job1_gpu_id_list,job2_gpu_id_list,shm_name_dict
            
    def execute_schedule(self, job, select_gpu_list, prior, shm_name_dict):
        world_size=0
        nprocs_list=[0]*self.master.node_num
        gpu_id_list=[ [] for i in range(self.master.node_num)]  #需要有顺序

        min_node_index=999    #选取最小的node index作为
        for [node_index, gpu_list] in select_gpu_list:
            
            world_size+=len(gpu_list)
            nprocs_list[node_index]=len(gpu_list)
            gpu_id_list[node_index]=gpu_list
            if len(gpu_list)>0:
                min_node_index=node_index if node_index<min_node_index else min_node_index
            
        main_ip=self.master.nodes[min_node_index].ip
        main_temp_port=self.master.nodes[min_node_index].get_idle_port()

        for [node_index, gpu_list] in select_gpu_list:
            if len(gpu_list)==0:    #如果对应GPU list没有被选择，则不用将Job发送到Node，不然，会导致任务重复
                continue
            job_t=copy.deepcopy(job)
            if node_index == min_node_index:
                job_t.set_is_main(True)
            else:
                job_t.set_is_main(False)

            net_card=self.master.nodes[node_index].net_card
            job_t.set_execute_info(main_ip, main_temp_port, net_card, node_index, world_size ,nprocs_list, gpu_id_list, prior=prior, shm_name_list=shm_name_dict)
            self.master.send_job_to_execution(job_t)
    
    
    
    
    # #over share 调度主线
    # def schedule_weave_over_sharing_back(self,job_list):
    #     multi_gpu_jobs, single_gpu_jobs=self.__over_sharing_classify_jobs(job_list)         #job 根据其并行数量（GPU数量）分类为多GPU任务和单GPU任务
    #     matched_multi_jobs_list=self.__over_sharing_match_jobs(multi_gpu_jobs)                    #对多GPU任务进行匹配
    #     matched_single_jobs_list=self.__over_sharing_match_jobs(single_gpu_jobs)                   #对单GPU任务进行匹配
        
    #     rest_jobs1=self.__over_sharing_select_gpu_and_execute_schedule(matched_multi_jobs_list)   #对多GPU任务， 选择GPU，并放到机器执行
    #     rest_jobs2=self.__over_sharing_select_gpu_and_execute_schedule(matched_single_jobs_list)   #对单GPU任务， 选择GPU，并放到机器执行
    #     rest_job=rest_jobs1+rest_jobs2                                                      #对未调度的任务，合并，并返回进行重调度
    #     return rest_job
    
         
    #     #选择合适的GPU，并发送Job执行
    # def __over_sharing_select_gpu_and_execute_schedule(self, matched_jobs_list):
    #     #matched_jobs_list = [job1, job2, [cpu, mem, gpu, gmem] ] or [job1, [cpu, mem, gpu, gmem] ] 
    #     rest_job=[]
    #     # scheduled_jobs=[]
    #     for matched_jobs in matched_jobs_list:
    #         if len(matched_jobs) == 3:
    #             job1=matched_jobs[0]
    #             job2=matched_jobs[1]
    #             pack_resource=matched_jobs[2]
    #         else:
    #             job1=matched_jobs[0]
    #             job2=None
    #             pack_resource=matched_jobs[1]
    #         #[[node_index, score, [[gpu_id, score],...]],...]
    #         satisfy_gpu_list=self.master.monitor.get_satisfy_gpu(pack_resource)
    #         #对优先级进行排序
    #         satisfy_gpu_list_new=[]
    #         all_satisfy_gpu_num=0
    #         for [node_index, score, temp_gpu_list] in satisfy_gpu_list:
    #             score=score+len(temp_gpu_list)*10-self.master.nodes[node_index].get_over_corss_num()*10
    #             #调整优先级，score原始是GPU剩余量百分比的和
    #             satisfy_gpu_list_new.append([node_index, score, temp_gpu_list])
    #             all_satisfy_gpu_num+=len(temp_gpu_list)
    #         satisfy_gpu_list_new.sort(key=lambda x:x[1], reverse=True)
            
            
    #         if job2 == None :
                
    #             max_parallel=job1.parallel_num
    #             if all_satisfy_gpu_num<max_parallel:
    #                 rest_job.append(job1)
    #                 continue
    #             job1_rest_parallel_num=job1.parallel_num
                
    #             job1_gpu_id_list, _, _= self.__over_sharing_select_gpu(job1_rest_parallel_num, 0, satisfy_gpu_list_new)   
    #             # scheduled_jobs.append([job1, job1_gpu_id_list, {}])
    #             [cpu, mem, gpu, gmem] =pack_resource
    #             job1.set_pack_resource(math.ceil(cpu), math.ceil(mem), math.ceil(gpu), math.ceil(gmem) )
    #             self.execute_schedule(job1, job1_gpu_id_list, False, {})
    #             self.master.monitor.alloc_resource(job1, None,job1_gpu_id_list)
    #             # self.execute_schedule(job1, job1_gpu_id_list, shm_name_dict)
                    
    #         else:
    #             max_parallel=max(job1.parallel_num, job2.parallel_num)
    #             if all_satisfy_gpu_num<max_parallel:
    #                 rest_job.append(job1)
    #                 rest_job.append(job2)
    #                 continue
    #             job1_rest_gpu=job1.parallel_num
    #             job2_rest_gpu=job2.parallel_num
                
    #             job1_gpu_id_list,job2_gpu_id_list,shm_name_dict= self.__over_sharing_select_gpu(job1_rest_gpu, job2_rest_gpu, satisfy_gpu_list_new)
    #             # scheduled_jobs.append([job1, job1_gpu_id_list, shm_name_dict])
    #             # scheduled_jobs.append([job2, job2_gpu_id_list, shm_name_dict])
    #             [cpu, mem, gpu, gmem] =pack_resource
    #             job1.set_pack_resource(math.ceil(cpu), math.ceil(mem), math.ceil(gpu), math.ceil(gmem), job2.job_name)
    #             job2.set_pack_resource(math.ceil(cpu), math.ceil(mem), math.ceil(gpu), math.ceil(gmem), job1.job_name)
    #             self.execute_schedule(job1, job1_gpu_id_list, True, shm_name_dict)
    #             self.execute_schedule(job2, job2_gpu_id_list, False,  shm_name_dict)
    #             if job1.parallel_num>= job2.parallel_num:
    #                 max_couple_job_gpu_id_list=job1_gpu_id_list
    #             else:
    #                 max_couple_job_gpu_id_list=job2_gpu_id_list
    #             self.master.monitor.alloc_resource(job1, job2,max_couple_job_gpu_id_list)
    #             # self.execute_schedule(job1, job1_gpu_id_list, shm_name_dict)
    #             # self.execute_schedule(job2, job2_gpu_id_list, shm_name_dict)
                
    #     return rest_job
            
                
        #     execute_node=None
            
        #     is_master_satisfy=
        #     #下面很多种情况，分开处理
        #     #两个node总和都无法完成任务
        #     if max_parallel>len(master_satisfy)+len(worker_satisfy):
        #         rest_job.append(job1)
        #         if job2 != None:
        #             rest_job.append(job2)
        #         continue
        #     #仅两个node合作可以完成
        #     if max_parallel<=len(master_satisfy)+len(worker_satisfy) and max_parallel>len(master_satisfy) and max_parallel>len(worker_satisfy):
        #         if self.master.monitor.cross_num>=self.master.max_cross:
        #             rest_job.append(job1)
        #             if job2 != None:
        #                 rest_job.append(job2)
        #             continue
        #         else:
        #             self.can_execute(master_satisfy, worker_satisfy, job1,job2)
        #             self.master.monitor.cross_num +=2
        #     #master上可以执行
        #     if max_parallel<=len(master_satisfy) and max_parallel>len(worker_satisfy):
        #         if self.master.monitor.cross_master_gpu_num>= self.master.max_gpu_cross:
        #             rest_job.append(job1)
        #             if job2 != None:
        #                 rest_job.append(job2)
        #             continue
        #         else:
        #             self.can_execute(master_satisfy, [], job1,job2)
        #             self.master.monitor.cross_master_gpu_num +=2
        #     #worker上可以执行
        #     if max_parallel>len(master_satisfy) and max_parallel<=len(worker_satisfy):
        #         if self.master.monitor.cross_worker_gpu_num>= self.master.max_gpu_cross:
        #             rest_job.append(job1)
        #             if job2 != None:
        #                 rest_job.append(job2)
        #             continue
        #         else:
        #             self.can_execute([], worker_satisfy, job1,job2)
        #             self.master.monitor.cross_worker_gpu_num +=2
            
            
        #     #两个都可以执行，需要选择
        #     if max_parallel<=len(master_satisfy) and max_parallel<=len(worker_satisfy):
        #         is_execute_master=True
        #         if self.master.monitor.cross_master_gpu_num >= self.master.max_gpu_cross and \
        #             self.master.monitor.cross_worker_gpu_num >= self.master.max_gpu_cross:
        #             rest_job.append(job1)
        #             if job2 != None:
        #                 rest_job.append(job2)
        #             continue
        #         elif self.master.monitor.cross_master_gpu_num >= self.master.max_gpu_cross:
        #             is_execute_master=False
        #         elif self.master.monitor.cross_worker_gpu_num < self.master.max_gpu_cross and\
        #             self.is_worker_proir(master_satisfy, worker_satisfy, max_parallel):
        #             is_execute_master=False
                
        #         if is_execute_master:
        #             self.can_execute(master_satisfy, [], job1,job2)
        #             self.master.monitor.cross_master_gpu_num +=2
        #         else:
        #             self.can_execute([], worker_satisfy, job1,job2)
        #             self.master.monitor.cross_worker_gpu_num +=2
        # return rest_job
    
    
    # def is_master_satisfy(self, master_satisfy_gpu_list, max_parrallel, job_num):
    #     if max_parrallel==1 :
    #         if len(master_satisfy_gpu_list)>0:
    #             return True
    #         else:
    #             return False
        
    #     if len(master_satisfy_gpu_list)>= max_parrallel and self.master.monitor.cross_master_gpu_num+job_num <= self.master.max_cross_gpu:
    #         return True
    #     else:
    #         return False

    # # def is_worker_satisfy()
          
    # def is_worker_prior(self,master_satisfy, worker_satisfy, max_parallel ):
    #         master_satisfy.sort(key=lambda x: x[1], reverse=True)
    #         worker_satisfy.sort(key=lambda x: x[1], reverse=True)
    #         master_v=0
    #         worker_v=0
    #         for i in range(max_parallel):
    #             master_v+=master_satisfy[i][1]
    #             worker_v+=worker_satisfy[i][1]
    #         if worker_v>master_v:
    #             return True
    #         else:
    #             return False
                    
    # def can_execute(self,master_satisfy, worker_satisfy, job1,job2):
    #     sync_name_dict={}
    #     selected1_gpu=self.select_proir_gpu(master_satisfy, worker_satisfy, job1)
    #     if job2 != None:
    #         selected2_gpu=self.select_proir_gpu(master_satisfy, worker_satisfy, job2)
    #         for gpu_id in selected2_gpu:
    #             if gpu_id in selected1_gpu:
    #                 shm_name=generate_shm_name()
    #                 sync_name_dict[gpu_id]=shm_name
    #         self.execute_schedule(job2, selected2_gpu, sync_name_dict)
    #     self.execute_schedule(job1, selected1_gpu, sync_name_dict)
        
                    
                    
            
    # def select_proir_gpu(master_satisfy,worker_satisfy, select_num):
    #     all_satisfy=master_satisfy[:]
    #     for worker_info in worker_satisfy:
    #         worker_info[0]=worker_info[0]+4
    #         all_satisfy.append(worker_info)
            
    #     all_satisfy.sort(key =lambda x : x[1], reverse=True)
    #     out_gpu_id=[]
    #     for i in range(select_num):
    #         out_gpu_id.append(master_satisfy[i][0])
    #     return out_gpu_id
        
            
            
    
    
    
    
    
    
    
    
    