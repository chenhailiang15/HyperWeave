import threading
import numpy as np
import math
class WeaveMonitor:
    def __init__(self,nodes, print_level):
        self.print_level=print_level
        self.nodes=nodes
        self.node_num=len(self.nodes)
        self.cross_maxtrix=np.zeros((self.node_num, self.node_num))
        self.gpu_num_list=[]
        for i in range(self.node_num):
            self.gpu_num_list.append(self.nodes[i].gpu_num)
            
        self.end_job_name=set()
        self.occupy_resource_gpu_id_list={}
        self.lock=threading.Lock()
        self.init_max_resource()
        
    

        
        
    def alloc_resource(self, instance1, instance2, couple_instance_gpu_id_list, plan=False):
        with self.lock:
            self.occupy_resource_gpu_id_list[instance1.instance_name]=couple_instance_gpu_id_list
            if plan==True:
                assert instance2 == None

                #plan resource
                if instance1.job.plan_gpu<=100:
                    for [node_index, gpu_id_list] in couple_instance_gpu_id_list:
                        self.nodes[node_index].alloc_resource(instance1.job.plan_cpu, instance1.job.plan_mem, instance1.job.plan_gpu, 0, gpu_id_list)
                else:
                    if instance1.job.plan_gpu%100 ==0 : #判断有没有GPU碎片的分配
                        fragement=False
                    else:
                        fragement=True


                    max_iter=len(couple_instance_gpu_id_list)
                    curr_iter_num=0
                    for [node_index, gpu_id_list] in couple_instance_gpu_id_list:
                        curr_iter_num += 1
                        if fragement == True and curr_iter_num==max_iter:
                            frage_part=(instance1.job.plan_gpu%100)/instance1.job.plan_gpu
                            self.nodes[node_index].alloc_resource(instance1.job.plan_cpu*frage_part, instance1.job.plan_mem*frage_part, instance1.job.plan_gpu*frage_part, 0, [gpu_id_list[0]])
                            self.nodes[node_index].alloc_resource(instance1.job.plan_cpu/instance1.job.plan_gpu/100, instance1.job.plan_mem/instance1.job.plan_gpu/100, 100, 0, gpu_id_list[1:])
                            fragement=False
                        else:
                            self.nodes[node_index].alloc_resource(instance1.job.plan_cpu/instance1.job.plan_gpu/100, instance1.job.plan_mem/instance1.job.plan_gpu/100, 100, 0, gpu_id_list)

                return
            else:
                #pack resource
                if instance2 !=None:
                    self.occupy_resource_gpu_id_list[instance2.instance_name]=couple_instance_gpu_id_list
                
                for [node_index, gpu_id_list] in couple_instance_gpu_id_list:
                    self.nodes[node_index].alloc_resource(instance1.pack_cpu, instance1.pack_mem, instance1.pack_gpu, instance1.pack_gmem, gpu_id_list)
        
                
    def takeback_resource(self,instance, plan=False):
        with self.lock:
            if instance.is_main == False:
                    return False
            couple_job_gpu_id_list=self.occupy_resource_gpu_id_list[instance.instance_name]
            if plan == True:
                #plan resource
                if instance.job.plan_gpu<=100:
                    for [node_index, gpu_id_list] in couple_job_gpu_id_list:
                        self.nodes[node_index].takeback_resource(instance.job.plan_cpu, instance.job.plan_mem, instance.job.plan_gpu, 0, gpu_id_list)
                else:
                    if instance.job.plan_gpu%100 ==0 : #判断有没有GPU碎片的分配
                        fragement=False
                    else:
                        fragement=True

                    max_iter=len(couple_job_gpu_id_list)
                    curr_iter_num=0
                    for [node_index, gpu_id_list] in couple_job_gpu_id_list:
                        curr_iter_num += 1
                        if fragement == True and curr_iter_num==max_iter:
                            frage_part=(instance.job.plan_gpu%100)/instance.job.plan_gpu
                            self.nodes[node_index].takeback_resource(instance.job.plan_cpu*frage_part, instance.job.plan_mem*frage_part, instance.job.plan_gpu*frage_part, 0, [gpu_id_list[0]])
                            self.nodes[node_index].takeback_resource(instance.job.plan_cpu/instance.job.plan_gpu/100, instance.job.plan_mem/instance.job.plan_gpu/100, 100, 0, gpu_id_list[1:])
                            fragement=False
                        else:
                            self.nodes[node_index].takeback_resource(instance.job.plan_cpu/instance.job.plan_gpu/100, instance.job.plan_mem/instance.job.plan_gpu/100, 100, 0, gpu_id_list)


            else:
                # pack resource
                if instance.couple_instance_name == None:   #没有耦合实例，则直接回收
                    for [node_index , gpu_id_list_t]in self.occupy_resource_gpu_id_list[instance.instance_name]:
                        self.nodes[node_index].takeback_resource(instance.pack_cpu, instance.pack_mem, instance.pack_gpu, instance.pack_gmem, gpu_id_list_t)
                    # del self.occupy_resource_gpu_id_list[job.job_name]
                    return
                if instance.couple_instance_name in self.end_job_name:  #有耦合实例，需要判断其是否已经结束
                    for [node_index , gpu_id_list_t] in self.occupy_resource_gpu_id_list[str(instance.job.job_idx)+"-"+str(instance.instance_idx)]:
                        self.nodes[node_index].takeback_resource(instance.pack_cpu, instance.pack_mem, instance.pack_gpu, instance.pack_gmem, gpu_id_list_t)
                    return
                else:
                    self.end_job_name.add(instance.instance_name)
        
        
    def get_satisfy_gpu(self,pack_resource=None):

        satisfy_gpu_list=[]
        for i in range(self.node_num):
            temp_gpu_list, score=self.nodes[i].get_satisfy_gpu_id(pack_resource)
            satisfy_gpu_list.append([i, score, temp_gpu_list])  #GPU数量最优先
        
        satisfy_gpu_list.sort(key=lambda x:x[1], reverse=True)    #对满足的node相关信息，进行排序，降序

        return satisfy_gpu_list        #[[node_index, score, [[gpu_id, score],...]],...]

    def get_satisfy_gpu_for_sim(self, pack_resource=None):
        satisfy_gpu_list = []
        for i in range(self.node_num):
            temp_gpu_list, score = self.nodes[i].get_satisfy_gpu_id_for_sim(pack_resource)
            satisfy_gpu_list.append([i, score, temp_gpu_list])  # GPU数量最优先

        satisfy_gpu_list.sort(key=lambda x: x[1], reverse=True)  # 对满足的node相关信息，进行排序，降序

        return satisfy_gpu_list  # [[node_index, score, [[gpu_id, score],...]],...]
    
    def init_max_resource(self):
        self.max_cpu=9600
        self.max_mem=512*1024
        self.max_gpu=800
        self.max_gmem=32*1024
        # for i in range(self.node_num):
        #     self.max_cpu=self.nodes[i].cpu if self.nodes[i].cpu>self.max_cpu else self.max_cpu
        #     self.max_mem=self.nodes[i].mem if self.nodes[i].mem>self.max_mem else self.max_mem
        #     self.max_gpu=self.nodes[i].gpu[0] if self.nodes[i].gpu[0]>self.max_gpu else self.max_gpu
        #     self.max_gmem=self.nodes[i].gmem[0] if self.nodes[i].gmem[0]>self.max_gmem else self.max_gmem

    def get_max_resource(self):
        return [self.max_cpu, self.max_mem, self.max_gpu, self.max_gmem]
    
    def get_idle_gpu_num(self):
        idel_gpu_num=0
        for node in self.nodes:
            idel_gpu_num+=node.get_idle_gpu_num()
        return idel_gpu_num


    def judge_runable_with_resource(self,resource):
        cpu_need=resource[0]
        mem_need=resource[1]
        gpu_need=resource[2]
        gmem_need=resource[3]

        if gpu_need <= 100:
            pack_resource = [cpu_need, mem_need, gpu_need, gmem_need]
            for i in range(self.node_num):
                satisfy_gpu_num = self.nodes[i].get_satisfy_gpu_num_by_cap(pack_resource)
                if satisfy_gpu_num>0:
                    return True

        else:
            pack_resource = [cpu_need/gpu_need/100, mem_need/gpu_need/100, 100, gmem_need/gpu_need/100]
            need_gpu_num=math.ceil(gpu_need/100)

            for i in range(self.node_num):
                satisfy_gpu_num = self.nodes[i].get_satisfy_gpu_num_by_cap(pack_resource)
                need_gpu_num-=satisfy_gpu_num
                if need_gpu_num<=0:
                    return True
        return False





    # def __get_satisfy_gpu_node(self,node_kind, pack_resource):
    #     cpu_need=pack_resource[0]
    #     mem_need=pack_resource[1]
    #     gpu_need=pack_resource[2]
    #     gmem_need=pack_resource[3]
    #     if node_kind=="master":
    #         master_satisfy=[]
    #         if self.master_cpu_rest<cpu_need or self.master_mem_rest<mem_need:
    #             return master_satisfy
    #         cpu_rest_per=(self.master_cpu_rest-cpu_need)/self.master_cpu
    #         mem_rest_per=(self.master_mem_rest-mem_need)/self.master_mem
            
    #         for i in range(4):
    #             if self.master_gpu_rest[i]>=gpu_need and self.master_gmem_rest[i]>=gmem_need:
    #                 gpu_rest_per=(self.master_gpu_rest[i]-gpu_need)/self.master_gpu[i]
    #                 gmem_rest_per=(self.master_gmem_rest[i]-gmem_need)/self.master_gmem[i]
    #                 ave_per=(cpu_rest_per+mem_rest_per+gpu_rest_per+gmem_rest_per)/4
    #                 master_satisfy.append([i,ave_per])
    #         return master_satisfy
    #     elif node_kind=="worker":
    #         worker_satisfy=[]
    #         if self.worker_cpu_rest<cpu_need or self.worker_mem_rest<mem_need:
    #             return worker_satisfy
    #         cpu_rest_per=(self.worker_cpu_rest-cpu_need)/self.worker_cpu
    #         mem_rest_per=(self.worker_mem_rest-mem_need)/self.worker_mem
            
    #         for i in range(4):
    #             if self.worker_gpu_rest[i]>=gpu_need and self.worker_gmem_rest[i]>=gmem_need:
    #                 gpu_rest_per=(self.worker_gpu_rest[i]-gpu_need)/self.worker_gpu[i]
    #                 gmem_rest_per=(self.worker_gmem_rest[i]-gmem_need)/self.worker_gmem[i]
    #                 ave_per=(cpu_rest_per+mem_rest_per+gpu_rest_per+gmem_rest_per)/4
    #                 worker_satisfy.append([i,ave_per])
    #         return worker_satisfy