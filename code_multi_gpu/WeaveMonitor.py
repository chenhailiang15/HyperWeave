import threading
import numpy as np

class WeaveMonitor:
    def __init__(self,nodes, print_level):
        self.print_level=print_level
        self.nodes=nodes
        self.node_num=len(self.nodes)
        self.cross_maxtrix=np.zeros((self.node_num, self.node_num))
        self.gpu_num_list=[]
        for i in range(self.node_num):
            self.gpu_num_list.append(self.nodes[i].gpu_num)
        
    

        
        
    def alloc_resource(self, job, job_gpu_id_list, pack_resource):
        [cpu, mem, gpu, gmem]=pack_resource
        for [node_index, gpu_id_list] in job_gpu_id_list:
            self.nodes[node_index].alloc_resource(cpu, mem, gpu, gmem, gpu_id_list)
        
                
    def takeback_resource(self,job, job_gpu_id_list, pack_resource):
        [cpu, mem, gpu, gmem]=pack_resource
        for [node_index, gpu_id_list] in job_gpu_id_list:
            self.nodes[node_index].takeback_resource(cpu, mem, gpu, gmem, gpu_id_list)
        
            
                
                
    def get_satisfy_gpu(self,pack_resource):
        satisfy_gpu_list=[]
        for i in range(self.node_num):
            temp_gpu_list, score=self.nodes[i].get_satisfy_gpu_id(pack_resource)
            satisfy_gpu_list.append([i, score, temp_gpu_list])  #GPU数量最优先
        
        # satisfy_gpu_list.sort(key=lambda x:x[1], reverse=True)#

        return satisfy_gpu_list
    
    def __get_satisfy_gpu_node(self,node_kind, pack_resource):
        cpu_need=pack_resource[0]
        mem_need=pack_resource[1]
        gpu_need=pack_resource[2]
        gmem_need=pack_resource[3]
        if node_kind=="master":
            master_satisfy=[]
            if self.master_cpu_rest<cpu_need or self.master_mem_rest<mem_need:
                return master_satisfy
            cpu_rest_per=(self.master_cpu_rest-cpu_need)/self.master_cpu
            mem_rest_per=(self.master_mem_rest-mem_need)/self.master_mem
            
            for i in range(4):
                if self.master_gpu_rest[i]>=gpu_need and self.master_gmem_rest[i]>=gmem_need:
                    gpu_rest_per=(self.master_gpu_rest[i]-gpu_need)/self.master_gpu[i]
                    gmem_rest_per=(self.master_gmem_rest[i]-gmem_need)/self.master_gmem[i]
                    ave_per=(cpu_rest_per+mem_rest_per+gpu_rest_per+gmem_rest_per)/4
                    master_satisfy.append([i,ave_per])
            return master_satisfy
        elif node_kind=="worker":
            worker_satisfy=[]
            if self.worker_cpu_rest<cpu_need or self.worker_mem_rest<mem_need:
                return worker_satisfy
            cpu_rest_per=(self.worker_cpu_rest-cpu_need)/self.worker_cpu
            mem_rest_per=(self.worker_mem_rest-mem_need)/self.worker_mem
            
            for i in range(4):
                if self.worker_gpu_rest[i]>=gpu_need and self.worker_gmem_rest[i]>=gmem_need:
                    gpu_rest_per=(self.worker_gpu_rest[i]-gpu_need)/self.worker_gpu[i]
                    gmem_rest_per=(self.worker_gmem_rest[i]-gmem_need)/self.worker_gmem[i]
                    ave_per=(cpu_rest_per+mem_rest_per+gpu_rest_per+gmem_rest_per)/4
                    worker_satisfy.append([i,ave_per])
            return worker_satisfy
                
            
    
    
    
    
    
