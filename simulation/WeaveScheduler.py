import math
import random
import os
import threading
from util import *
import copy
import time
from blossom import Blossom_Same


class WeaveSchedulor:
    def __init__(self,master, print_level=0):
        
        self.master=master
        self.strategy=master.schedule_strategy
        self.gpu_list=[i for i in range(7)]
        self.print_level=print_level
        
    
    def do_schedule(self,job_list):
        # if self.print_level>2:
        #     print(f"start schedule ({self.strategy})...")
            
        if self.master.system=="Weave":
            rest_job=self.schedule_weave(job_list)
        elif self.master.system=="Muri":
            rest_job=self.schedule_muri(job_list)
        elif self.master.system == "Normal":
            rest_job=self.schedule_normal(job_list)
        else:
            print(f"system name wrong! {self.master.system} (should be Weave, Muri or Normal)")
            
        return rest_job
    
    #Weave 调度主线
    def schedule_weave(self,job_list):
        # if self.master.env.now>3038583:
        #     a=0
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
    
    
    
    
    
    
    #muri 调度主线
    def schedule_muri(self,job_list):
        self.job_idx_to_job={}
        all_matched_job_list=[]
        job_group={}
        match_job_num=0
        #将Job根据GPU使用数量打包
        for job in job_list:
            self.job_idx_to_job[job.job_idx]=job
            job_mini={}
            job_mini["num_gpu"]=job.parallel_num
            job_mini['resource_time']=self.master.analyze_loader.get_time_all(job.get_name_batchsize_epoch())
            job_mini['job_idx']=job.job_idx
            job_mini['iteration_time']=job.total_epochs
            
            if job.parallel_num in job_group:
                job_group[job.parallel_num].append(job_mini)
            else:
                job_group[job.parallel_num]=[job_mini]
        
        packings=Blossom_Same.run(job_group, self.master.monitor.get_idle_gpu_num())
        
        succeed_matched_job_idx=set()
        faile_matched_job_idx=set()
        for gpu_num in packings:
            # print(f"blossom gpu num:{gpu_num} ... ")
            
            for _pack in packings[gpu_num]:    #obj _pack : _Packing  这里面每个循环是一个匹配
                matched_job=[]
                matched_flage=True
                # print(f"one pack:\t", end="")
                for job_t in _pack.best_permutation: #这里总共是一个匹配
                    # print(job_t.job_idx,end="\t")
                    matched_job.append(self.job_idx_to_job[job_t.job_idx]) 
                    #如果有一个已经属于被匹配了的，本次匹配失败，后续单独处理
                    if job_t.job_idx in succeed_matched_job_idx:
                        matched_flage=False
                # print("")
            
            
                #处理一个匹配
                if matched_flage==True:
                    if self.is_max_resource_satisfy(matched_job): #判断资源是否足够，足够才能算匹配成功
                        all_matched_job_list.append(matched_job)
                        match_job_num+=len(matched_job)
                        # print(f"0000000000000000000000000000000000000000000000000 match num:  {len(matched_job)}")
                        for job in matched_job:
                            succeed_matched_job_idx.add(job.job_idx)
                    else:
                        for job in matched_job:
                            faile_matched_job_idx.add(job.job_idx)
                else:
                    for job in matched_job:
                        for job in matched_job:
                            faile_matched_job_idx.add(job.job_idx)
        #将失败的单独调度
        for job_idx in faile_matched_job_idx:
            if job_idx not in succeed_matched_job_idx:
                all_matched_job_list.append([self.job_idx_to_job[job_idx]])
                match_job_num+=1
        try:     
            assert match_job_num==len(job_list)
        except:
            for job in job_list:
                if job.job_idx not in faile_matched_job_idx and job.job_idx not in succeed_matched_job_idx:
                    print(f"******************fix blossom with add job:{job.job_idx}")
                    all_matched_job_list.append([job])
        
        if self.strategy=="FIFO":
            rest_job=self.schedule_muri_FIFO(all_matched_job_list)
        elif self.strategy=="SRTF":
            rest_job=self.schedule_muri_SRTF(all_matched_job_list)
        elif self.strategy=="SRSF":
            rest_job=self.schedule_muri_SRSF(all_matched_job_list)
        else:
            print("strategy wrong!")
            exit(-1)

        return rest_job
    
    def is_max_resource_satisfy(self, matched_job):
        max_mem=0
        max_gmem=0
        for job in matched_job:
            max_mem+=self.master.analyze_loader.get_job_pack_resource(job)[1]
            max_gmem+=self.master.analyze_loader.get_job_pack_resource(job)[3]
            
        if max_mem<= self.master.monitor.max_mem and max_gmem<= self.master.monitor.max_gmem:
            print(f"匹配好了，内存资源够！True True True {len(matched_job)}")
            return True
        else:
            print(f"匹配好了，但是内存资源不够！False False False {len(matched_job)}")
            return False
        
        
        
    
    def schedule_muri_FIFO(self, job_matched_end):
        order_matched_jobs=[]
        for job_list in job_matched_end:
            arrive_time=float('inf')
            for job in job_list:
                arrive_time=min(arrive_time, job.arrive_time)
            order_matched_jobs.append([arrive_time, job_list])
        #matched_jobs排序
        order_matched_jobs.sort(key=lambda x : x[0])
        rest_job=self.schedule_muri_ordered_matched_job_list(order_matched_jobs)
        return rest_job
    
    def schedule_muri_SRTF(self, job_matched_end):
        order_matched_jobs=[]
        time_now=time.time()
        for job_list in job_matched_end:
            rest_time=float('inf')
            for job in job_list:
                rest_time_t=job.ddl_time-time_now-job.duration_time
                rest_time=min(rest_time, rest_time_t)
            order_matched_jobs.append([rest_time, job_list])
        #matched_jobs排序
        order_matched_jobs.sort(key=lambda x : x[0])
        rest_job=self.schedule_muri_ordered_matched_job_list(order_matched_jobs)
        return rest_job
    
    def schedule_muri_SRSF(self, job_matched_end):
        order_matched_jobs=[]
        time_now=time.time()
        for job_list in job_matched_end:
            rest_time=float('inf')
            for job in job_list:
                rest_time_t=(job.ddl_time-time_now-job.duration_time)**job.parallel_num
                rest_time=min(rest_time, rest_time_t)
            order_matched_jobs.append([rest_time, job_list])
        #matched_jobs排序
        order_matched_jobs.sort(key=lambda x : x[0])
        rest_job=self.schedule_muri_ordered_matched_job_list(order_matched_jobs)
        return rest_job
    
    def schedule_muri_ordered_matched_job_list(self, match_jobs):
        #初始化未被调度的job
        rest_job=[]
        #是否继续调度的标志，当遇到一个无法调度的任务时，停止调度等待下一轮调度，将剩余的job返回
        continue_schedule_flage=True
        #循环调度
        for [order_value, job_list] in match_jobs:
            if continue_schedule_flage:
                #正常调度
                need_gpu_num = job_list[0].parallel_num
                
                #[[node_index, score, [[gpu_id, score],...]],...]
                satisfy_gpu_list=self.master.monitor.get_satisfy_gpu()
                #对优先级进行排序
                
                all_satisfy_gpu_num=0
                for [node_index, score, temp_gpu_list] in satisfy_gpu_list:
                    all_satisfy_gpu_num+=len(temp_gpu_list)
                
                if all_satisfy_gpu_num< need_gpu_num:
                    rest_job.extend(job_list)
                    continue_schedule_flage=False
                    continue
                
                job_gpu_id_list=[]
                shm_name_dict={}
                
                for [node_index, score, temp_gpu_list] in satisfy_gpu_list:
                    job_temp_gpu_id_list=[]
                    shm_name_dict_temp={}
                    for gpu_id in temp_gpu_list:
                        
                        if need_gpu_num>0:
                            job_temp_gpu_id_list.append(gpu_id)
                            shm_name=generate_shm_name()
                            shm_name_dict_temp[gpu_id]=shm_name
                            
                            need_gpu_num-=1
                    job_gpu_id_list.append([node_index, job_temp_gpu_id_list])    
                    shm_name_dict[node_index]=shm_name_dict_temp
                
                idx_on_gou=0
                
                if len(job_list)==1:
                    shm_name_dict={}
                for job in job_list:
                    job.idx_on_gou=idx_on_gou
                    job.max_sync_num=len(job_list)
                    idx_on_gou+=1
                    print(f"start do job:{job.job_idx}")
                    self.master.monitor.alloc_resource(job, None,job_gpu_id_list, plan=True)
                    self.execute_schedule(job, job_gpu_id_list, False, shm_name_dict)
            else:
                rest_job.extend(job_list)
                
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
                    if self.master.weave_sync_mode==False:
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
                if job1.job_name =="37c4f6f6faf7b828eb52cb80" or job2.job_name=="37c4f6f6faf7b828eb52cb80":
                    a=9
                
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
            
    def execute_schedule(self, instance, select_gpu_list, prior, shm_name_dict):
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
            instance_t=copy.deepcopy(instance)
            instance_t.job=instance.job
            if node_index == min_node_index:
                instance_t.set_is_main(True)
            else:
                instance_t.set_is_main(False)
        #
            net_card=self.master.nodes[node_index].net_card
            instance_t.set_execute_info(main_ip, main_temp_port, net_card, node_index, world_size ,nprocs_list, gpu_id_list, prior=prior, shm_name_list=shm_name_dict)
            self.master.send_instance_to_execution(instance_t)
    
    
    
    #********************************************************************************Normal***********************************************************************************************
    #normal 调度主线
    def schedule_normal(self,job_list):
        if self.strategy=="FIFO":
            rest_job=self.schedule_normal_FIFO(job_list)
        elif self.strategy=="SRTF":
            rest_job=self.schedule_normal_SRTF(job_list)
        elif self.strategy=="SRSF":
            rest_job=self.schedule_normal_SRSF(job_list)
        else:
            print("strategy wrong!")
            exit(-1)

        return rest_job
    
    
    
    def schedule_normal_FIFO(self, job_list):
        #按照到来的先后顺序排序
        job_list.sort(key=lambda x: x.arrive_time)
        rest_job=self.schedule_normal_single_job_list(job_list)
            
        return rest_job
    
    def schedule_normal_SRTF(self, job_list):
        time_now=time.time()
        #按照到来的先后顺序排序
        job_list.sort(key=lambda x: x.ddl_time-time_now-x.duration_time)
        rest_job=self.schedule_normal_single_job_list(job_list)
            
        return rest_job
    
    def schedule_normal_SRSF(self, job_list):
        
        time_now=time.time()
        # for job in job_list:
        #     job.set_schedule_order(time_now)
            
        #按照到来的先后顺序排序
        job_list.sort(key=lambda x: (x.ddl_time-time_now-x.duration_time)*x.parallel_num)
        rest_job=self.schedule_normal_single_job_list(job_list)
        return rest_job


    def schedule_normal_single_job_list(self, job_list):
        #初始化未被调度的job
        rest_job=[]
        #是否继续调度的标志，当遇到一个无法调度的任务时，停止调度等待下一轮调度，将剩余的job返回
        continue_schedule_flage=True
        #循环调度
        for job in job_list:
            if continue_schedule_flage:
                #正常调度
                for index in range(len(job.instance_list)):

                    if job.plan_gpu<=100:
                        plan_resource=[job.plan_cpu, job.plan_mem, job.plan_gpu, 0]
                    else:
                        plan_resource = [job.plan_cpu / job.plan_gpu / 100, job.plan_mem / job.plan_gpu / 100, 100, 0]

                    satisfy_gpu_list=self.master.monitor.get_satisfy_gpu_for_sim(plan_resource)
                    all_satisfy_gpu_num=0

                    for [node_index, score, temp_gpu_list] in satisfy_gpu_list:
                        all_satisfy_gpu_num+=len(temp_gpu_list)
                    if all_satisfy_gpu_num>=job.parallel_num:
                        selected_gpu_id_list,_,_= self.__over_sharing_select_gpu(job.parallel_num, 0, satisfy_gpu_list)
                        instance=job.instance_list.pop(0)
                        job.dealing_instance_num+=1
                        self.master.monitor.alloc_resource(instance, None, selected_gpu_id_list, plan=True)
                        self.execute_schedule(instance, selected_gpu_id_list, False, {})
                    else:
                        rest_job.append(job)
                        continue_schedule_flage=False
            else:
                rest_job.append(job)
        return rest_job

    # def place_jobs_to_nodes(self, job_list, mode="plan") :
    #
    #
    #
    #     job1_gpu_id_list=[]
    #     job2_gpu_id_list=[]
    #     shm_name_dict={}
    #     for [node_index, score, temp_gpu_list] in satisfy_gpu_list_new:
    #         job1_temp_gpu_id_list=[]
    #         job2_temp_gpu_id_list=[]
    #         shm_name_dict_temp={}
    #         for [gpu_index, ave_per] in temp_gpu_list:
    #             if job1_rest_gpu>0 and job2_rest_gpu>0:
    #                 shm_name=generate_shm_name()
    #                 shm_name_dict_temp[gpu_index]=shm_name
    #                 job1_temp_gpu_id_list.append(gpu_index)
    #                 job2_temp_gpu_id_list.append(gpu_index)
    #                 job1_rest_gpu-=1
    #                 job2_rest_gpu-=1
    #             elif job1_rest_gpu>0 and job2_rest_gpu==0:
    #                 job1_temp_gpu_id_list.append(gpu_index)
    #                 job1_rest_gpu-=1
    #             elif job1_rest_gpu==0 and job2_rest_gpu>0:
    #                 job2_temp_gpu_id_list.append(gpu_index)
    #                 job2_rest_gpu-=1
    #             elif job1_rest_gpu==0 and job2_rest_gpu==0:
    #                 break
    #             else:
    #                 print("(monitor) wrong!")
    #                 exit(256)
    #         job1_gpu_id_list.append([node_index, job1_temp_gpu_id_list])
    #         job2_gpu_id_list.append([node_index, job2_temp_gpu_id_list])
    #         shm_name_dict[node_index]=shm_name_dict_temp
    #         if job1_rest_gpu==0 and job2_rest_gpu==0:
    #             break
    #     return job1_gpu_id_list,job2_gpu_id_list,shm_name_dict
        
        
    