import json
import time
import math
from util import *



class Job:
    def __init__(self,job_idx, system):
        self.arrive_time=0
        self.start_time=0
        self.end_time=0
        self.succeed_flage=None
        self.job_idx=job_idx
        self.idx_on_gpu=0
        self.system=system
        self.max_sync_num=0

        self.batch_num=100

        self.time_init=0
        self.time_init_iter=0
        self.time_get_data=0
        self.time_forward_back=0
        self.time_commu=0
        self.time_epoch_no_init_iter =0

        self.used_resource_cpu=[0,0,0]
        self.used_resource_mem=[0,0,0]
        self.used_resource_gpu=[0,0,0]
        self.used_resource_gmem=[0,0,0]

        self.instance_list = []
        self.instance_num=0
        self.dealing_instance_num=0
        self.succeed_instance_num=0
        self.failed_instance_num=0
        self.time_extend_list={}   #  instance_index->[[time, gpu_id, new_number],]  
        self.time_extend_store={}
        self.init_iter_percent=0     #job特性，init iter耗时所占总epoch的时间比例

    def __eq__(self, other):
        if isinstance(other, Job):
            return self.job_idx == other.job_idx
        return False
        
    #模型信息        
    def set_model_info(self,job_name, model_name,total_epochs, batch_size, worker_num=4, layer_num=10, layer_feature=10,squad_data_size=1000 ):
        self.job_name=job_name
        self.model_name=model_name
        self.total_epochs=total_epochs
        self.batch_size=batch_size
        self.worker_num=worker_num
        self.layer_num=layer_num
        self.layer_feature=layer_feature
        self.squad_data_size=squad_data_size
    
    #计划资源
    def set_plan_resource(self, plan_cpu, plan_mem, plan_gpu):
        self.plan_cpu=plan_cpu
        self.plan_mem=plan_mem
        self.plan_gpu=plan_gpu
        self.parallel_num=math.ceil(plan_gpu/100)
        
    # def set_pack_resource(self, pack_cpu, pack_mem, pack_gpu, pack_gmem, couple_job_name=None):
    #     self.pack_cpu=pack_cpu
    #     self.pack_mem=pack_mem
    #     self.pack_gpu=pack_gpu
    #     self.pack_gmem=pack_gmem
    #     self.couple_job_name=couple_job_name
        
    # def set_is_main(self, is_main):
    #     self.is_main=is_main
    
    # def set_gpu_list(self, gpu_list):
    #     self.gpu_list=gpu_list
    def set_time_for_muri(self):
        self.four_times=self.get_four_stage_time()
    
    #时间信息
    def set_arrive_time(self, arrive_time):
        self.arrive_time=arrive_time

    def set_start_time(self, start_time):
        self.start_time=start_time

    def set_end_time(self, end_time):
        self.end_time=end_time

    def set_ddl_time(self, ddl_time):
        self.ddl_time=ddl_time

    def set_duration_time(self, duration_time):
        self.duration_time=duration_time
        
    def set_schedule_order(self,time_now):
        self.order= (self.ddl_time-time_now-self.duration_time)*self.parallel_num
    
    def succeed(self):
        self.succeed_flage=True

    def failed(self):
        self.succeed_flage=False

    def set_pack_resource(self, pack_cpu, pack_mem, pack_gpu, pack_gmem, couple_job_name=None):
        self.pack_cpu = pack_cpu
        self.pack_mem = pack_mem
        self.pack_gpu = pack_gpu
        self.pack_gmem = pack_gmem
        self.couple_job_name = couple_job_name
        
        
    def job_key_info(self):
        return f"jn:{self.job_name}-mn:{self.model_name}-epo:{self.total_epochs}-bts:{self.batch_size}-gpu:{self.plan_gpu}"
    
    def get_name_batchsize_epoch(self): #这里主要用于从分析中拿取数据，因此最大并行度为4
        parallel_num=min(self.parallel_num, 4)# 后续可能需要调整
        return f"{self.model_name}-{self.batch_size}-{parallel_num}"

    def get_three_stage_time(self):
        return [self.time_init, self.time_init_iter, (self.time_get_data+self.time_forward_back+ self.time_commu)*self.batch_num-self.time_get_data]

    def get_four_stage_time(self):
        return [self.time_init, self.time_get_data, self.time_forward_back, self.time_commu]

    def get_max_used_resource(self):
        return [max(self.used_resource_cpu), max(self.used_resource_mem), max(self.used_resource_gpu), max(self.used_resource_gmem)]

    def get_plan_resource(self):
        return [self.plan_cpu, self.plan_mem, self.plan_gpu]
        
    def is_multi_gpu(self):
        if self.plan_gpu>100:
            return True
        else:
            return False
        
    def get_time_extend(self, instance_idx, time_start, time_now):
        # self.time_extend_list={}   #  instance_index->[[time, gpu_id, new_number],]  
        # self.time_extend_store={}
        
        if instance_idx in self.time_extend_store and time_now in self.time_extend_store[instance_idx]:
            return self.time_extend_store[instance_idx][time_now]

        time_extend=0
        temp_dict={}
        
        global_max_parallel=0
        global_max_parallel_gpu_id=None
        max_parallel_start_time=time_start
        last_element=None
        
        
        self.time_extend_list[instance_idx].sort(key=lambda x:x[0], reverse=True)
        init_len=len(self.time_extend_list[instance_idx])
        
        
        for index in range(len(self.time_extend_list[instance_idx])-1,-1,-1):
            [time_t, gpu_id, cur_num]=self.time_extend_list[instance_idx][index]
            if index == init_len-1:  #并行数量初始化
                
                assert time_t < time_start or abs(time_t-time_start)<0.00001
                
                global_max_parallel=cur_num
                global_max_parallel_gpu_id=gpu_id
                last_element=self.time_extend_list[instance_idx][index]
                
            #更新当前GPU最大并行    
            if gpu_id not in  temp_dict:
                temp_dict[gpu_id]=cur_num
            elif cur_num>temp_dict[gpu_id]:
                temp_dict[gpu_id]=cur_num
            
            
            local_max_parallel_gpu_id, local_max_parallel=max(temp_dict.items(), key=lambda item: item[1]) 
            # if local_max_parallel==global_max_parallel and local_max_parallel_gpu_id!= global_max_parallel_gpu_id:
            #     self.time_extend_list[instance_idx].remove(last_element)
            #     last_element=self.time_extend_list[instance_idx][index]
                
                
            #判断是否需要更新time_extend
            if time_t> max_parallel_start_time and local_max_parallel!=global_max_parallel:

                time_extend+=(mps_time_extend[self.model_name][global_max_parallel-1]-1)*(time_t-max_parallel_start_time)
                global_max_parallel=local_max_parallel
                max_parallel_start_time=time_t
                
                self.time_extend_list[instance_idx].remove(last_element)
                last_element=self.time_extend_list[instance_idx][index]
            #更新最大值对应的GPU ID，防止后续删除错误
            elif local_max_parallel==global_max_parallel and local_max_parallel_gpu_id != global_max_parallel_gpu_id:
                self.time_extend_list[instance_idx].remove(last_element)
                last_element=self.time_extend_list[instance_idx][index]
            else:
                if self.time_extend_list[instance_idx][index]!=last_element:
                    del self.time_extend_list[instance_idx][index]
        
        time_extend=round(time_extend, 6)
        if instance_idx not in self.time_extend_store:
            self.time_extend_store[instance_idx]={}
            self.time_extend_store[instance_idx][time_now]=time_extend
        else:
            self.time_extend_store[instance_idx][time_now]=time_extend
            
        return time_extend


class Instance:
    def __init__(self,job, instance_idx, system):
        self.job=job
        self.instance_name=str(job.job_idx)+"-"+str(instance_idx)
        self.start_time = 0
        self.end_time = 0
        self.succeed_flage = None
        self.instance_idx = instance_idx
        self.instance_global_idx=0
        self.idx_on_gpu = 0
        self.system = system
        self.max_sync_num = 0

        self.batch_num = 100
        self.rest_batch_num=0
        self.init_iter_num=0
        

    def __eq__(self, other):
        if isinstance(other, Instance):
            return self.instance_global_idx == other.instance_global_idx
        return False



    def set_pack_resource(self, pack_cpu, pack_mem, pack_gpu, pack_gmem, couple_instance_name=None):
        self.pack_cpu = pack_cpu
        self.pack_mem = pack_mem
        self.pack_gpu = pack_gpu
        self.pack_gmem = pack_gmem
        self.couple_instance_name = couple_instance_name

    def set_is_main(self, is_main):
        self.is_main = is_main





    def set_start_time(self, start_time):
        self.start_time = start_time

    def set_end_time(self, end_time):
        self.end_time = end_time

    def set_duration_time(self, duration_time):
        self.duration_time=duration_time

    def set_schedule_order(self, time_now):
        self.order = (self.ddl_time - time_now - self.duration_time) * self.parallel_num

    def succeed(self):
        self.succeed_flage = True

    def failed(self):
        self.succeed_flage = False

    def set_execute_info(self, MASTER_ADDR, MASTER_PORT, net_card, node_rank, world_size, nprocs_list, gpu_id_list, prior=False, shm_name_list={}):
        self.MASTER_ADDR = MASTER_ADDR
        self.MASTER_PORT = MASTER_PORT
        self.net_card = net_card
        self.node_rank = node_rank
        self.world_size = world_size
        self.nprocs_list = nprocs_list
        self.gpu_id_list = gpu_id_list
        self.prior = prior
        self.shm_name_list = shm_name_list


    def job_key_info(self):
        return f"jn:{self.job_name}-mn:{self.model_name}-tep:{self.total_epochs}-bts:{self.batch_size}-gpu:{self.plan_gpu}"

    def get_name_batchsize_epoch(self):  # 这里主要用于从分析中拿取数据，因此最大并行度为4
        parallel_num = min(self.parallel_num, 4)  # 后续可能需要调整
        return f"{self.model_name}-{self.batch_size}-{parallel_num}"

    def is_only_master(self):
        if self.world_size == self.nprocs_list[0]:
            return True
        else:
            return False

    def is_only_worker(self):
        if self.world_size == self.nprocs_list[1]:
            return True
        else:
            return False

    def is_cross(self):
        if self.nprocs_list[0] != 0 and self.nprocs_list[1] != 0:
            return True
        else:
            return False

    def is_multi_gpu(self):
        if self.plan_gpu > 100:
            return True
        else:
            return False
        
        
    def get_time_extend(self, time_start, time_now):
        return self.job.get_time_extend(self.instance_idx, time_start, time_now)