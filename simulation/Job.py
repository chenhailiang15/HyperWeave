import json
import time
import math




class Job:
    def __init__(self,job_idx, system):
        self.arrive_time=0
        self.start_time=0
        self.end_time=0
        self.succeed_flage=None
        self.job_idx=job_idx
        self.idx_on_gou=0
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



class Instance:
    def __init__(self,job, instance_idx, system):
        self.job=job
        self.instance_name=str(job.job_idx)+"-"+str(instance_idx)
        self.start_time = 0
        self.end_time = 0
        self.succeed_flage = None
        self.instance_idx = instance_idx
        self.instance_global_idx=0
        self.idx_on_gou = 0
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

        # nprocs_list_c = nprocs_list.__str__().replace(" ", "")
        # gpu_id_list_c = gpu_id_list.__str__().replace(" ", "")
        # shm_name_list_c = json.dumps(shm_name_list).replace(" ", "")
        # shm_name_list_c = "\"" + str(shm_name_list) + "\""
        # # print("(job) shm_name_list_c dict:", shm_name_list_c)
        #
        # self.command = f"python WeaveExecutor.py --MASTER_ADDR {MASTER_ADDR} --MASTER_PORT {MASTER_PORT} --net_card {net_card}  --node_rank {node_rank} \
        #     --world_size {world_size} --nprocs_list {nprocs_list_c} --gpu_id_list {gpu_id_list_c} --model_name {self.job.model_name} --batch_size {self.batch_size} \
        #     --batch_num {self.job.batch_num} --total_epochs {self.job.total_epochs} --worker_num {self.job.worker_num} --layer_num {self.job.layer_num} --layer_feature {self.job.layer_feature} \
        #     --squad_data_size {self.job.squad_data_size} --shm_name_list {shm_name_list_c} --job_idx {self.job_idx} --idx_on_gpu {self.idx_on_gou} --system {self.system} \
        #     --max_sync_num {self.max_sync_num}"
        # if prior:
        #     self.command = self.command + " --prior"

    def to_string(self):
        json_dict = {}

        json_dict["job_idx"] = self.job_idx
        json_dict["idx_on_gou"] = self.idx_on_gou
        json_dict["system"] = self.system

        json_dict["job_name"] = self.job_name
        json_dict["model_name"] = self.model_name
        json_dict["total_epochs"] = self.total_epochs
        json_dict["batch_size"] = self.batch_size
        json_dict["batch_num"] = self.batch_num
        json_dict["worker_num"] = self.worker_num
        json_dict["layer_num"] = self.layer_num
        json_dict["layer_feature"] = self.layer_feature
        json_dict["squad_data_size"] = self.squad_data_size

        json_dict["MASTER_ADDR"] = self.MASTER_ADDR
        json_dict["MASTER_PORT"] = self.MASTER_PORT
        json_dict["net_card"] = self.net_card
        json_dict["node_rank"] = self.node_rank
        json_dict["world_size"] = self.world_size
        json_dict["nprocs_list"] = self.nprocs_list
        json_dict["gpu_id_list"] = self.gpu_id_list
        json_dict["prior"] = self.prior
        json_dict["shm_name_list"] = self.shm_name_list
        json_dict["command"] = self.command

        json_dict["plan_cpu"] = self.plan_cpu
        json_dict["plan_mem"] = self.plan_mem
        json_dict["plan_gpu"] = self.plan_gpu
        json_dict["parallel_num"] = self.parallel_num

        json_dict["pack_cpu"] = self.pack_cpu
        json_dict["pack_mem"] = self.pack_mem
        json_dict["pack_gpu"] = self.pack_gpu
        json_dict["pack_gmem"] = self.pack_gmem
        json_dict["couple_job_name"] = self.couple_job_name

        json_dict["is_main"] = self.is_main
        # json_dict["gpu_list"]=self.gpu_list

        json_dict["arrive_time"] = self.arrive_time
        json_dict["start_time"] = self.start_time
        json_dict["end_time"] = self.end_time
        json_dict["ddl_time"] = self.ddl_time
        json_dict["duration_time"] = self.duration_time
        json_dict["succeed_flage"] = self.succeed_flage

        return json.dumps(json_dict)

    def load_string(self, json_string):
        json_dict = json.loads(json_string)

        self.job_idx = json_dict["job_idx"]
        self.idx_on_gou = json_dict["idx_on_gou"]
        self.system = json_dict["system"]

        self.job_name = json_dict["job_name"]
        self.model_name = json_dict["model_name"]
        self.total_epochs = json_dict["total_epochs"]
        self.batch_size = json_dict["batch_size"]
        self.batch_num = json_dict["batch_num"]
        self.worker_num = json_dict["worker_num"]
        self.layer_num = json_dict["layer_num"]
        self.layer_feature = json_dict["layer_feature"]
        self.squad_data_size = json_dict["squad_data_size"]

        self.MASTER_ADDR = json_dict["MASTER_ADDR"]
        self.MASTER_PORT = json_dict["MASTER_PORT"]
        self.net_card = json_dict["net_card"]
        self.node_rank = json_dict["node_rank"]
        self.world_size = json_dict["world_size"]
        self.nprocs_list = json_dict["nprocs_list"]
        self.gpu_id_list = json_dict["gpu_id_list"]
        self.prior = json_dict["prior"]
        self.shm_name_list = json_dict["shm_name_list"]  # 由于是字典，这里需要调整key的类型。从string转换为 int （暂时不影响，暂不修改）
        self.command = json_dict["command"]

        self.plan_cpu = json_dict["plan_cpu"]
        self.plan_mem = json_dict["plan_mem"]
        self.plan_gpu = json_dict["plan_gpu"]
        self.parallel_num = json_dict["parallel_num"]

        self.pack_cpu = json_dict["pack_cpu"]
        self.pack_mem = json_dict["pack_mem"]
        self.pack_gpu = json_dict["pack_gpu"]
        self.pack_gmem = json_dict["pack_gmem"]
        self.couple_job_name = json_dict["couple_job_name"]

        self.is_main = json_dict["is_main"]
        # self.gpu_list=json_dict["gpu_list"]

        self.arrive_time = json_dict["arrive_time"]
        self.start_time = json_dict["start_time"]
        self.end_time = json_dict["end_time"]
        self.ddl_time = json_dict["ddl_time"]
        self.duration_time = json_dict["duration_time"]
        self.succeed_flage = json_dict["succeed_flage"]

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