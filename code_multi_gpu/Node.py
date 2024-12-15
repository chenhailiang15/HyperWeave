from util import *
import numpy as np


class Node:
    def __init__(self, node_id, ip,overshared_factor, max_cross_gpu_job_num, print_level):
        self.node_id=node_id
        self.overshared_factor=overshared_factor
        self.max_cross_gpu_job_num=max_cross_gpu_job_num
        self.print_level=print_level
        self.cross_gpu_job_num=0
        self.current_port=2000
        self.ip=ip
        
    def set_init_resouce(self, cpu, mem, gpu_num, gmem):
        self.cpu=cpu
        self.mem=mem
        self.gpu=np.array([100*self.overshared_factor for i in range(gpu_num)])
        self.gmem=np.array([gmem for i in range(gpu_num)])
        self.gpu_num=gpu_num
        
        self.cpu_rest=cpu
        self.mem_rest=mem
        self.gpu_rest=np.array([100*self.overshared_factor for i in range(gpu_num)])
        self.gmem_rest=np.array([gmem for i in range(gpu_num)])
        
    def alloc_resource(self, cpu, mem, gpu, gmem, gpu_id_list):
        self.cpu_rest-=cpu
        self.mem_rest-=mem
        for gpu_id in gpu_id_list:
            self.gpu_rest[gpu_id]=self.gpu_rest[gpu_id]-gpu
            self.gmem_rest[gpu_id]=self.gmem_rest[gpu_id]-gmem
        
    def takeback_resource(self, cpu, mem, gpu, gmem, gpu_id_list):
        self.cpu_rest+=cpu
        self.mem_rest+=mem
        for gpu_id in gpu_id_list:
            self.gpu_rest[gpu_id]=self.gpu_rest[gpu_id]+gpu
            self.gmem_rest[gpu_id]=self.gmem_rest[gpu_id]+gmem
            
    def get_satisfy_gpu_id(self,pack_resource):
        cpu_need=pack_resource[0]
        mem_need=pack_resource[1]
        gpu_need=pack_resource[2]
        gmem_need=pack_resource[3]
        satisfy_gpu_id_list=[]
        satisfy_score=0
        if self.cpu_rest<cpu_need or self.mem_rest<mem_need:
            return satisfy_gpu_id_list
        cpu_rest_per=(self.cpu_rest-cpu_need)/self.cpu
        mem_rest_per=(self.mem_rest-mem_need)/self.mem
        
        for i in range(self.gpu_num):
            if self.gpu_rest[i]>=gpu_need and self.gmem_rest[i]>=gmem_need:
                gpu_rest_per=(self.gpu_rest[i]-gpu_need)/self.gpu[i]
                gmem_rest_per=(self.gmem_rest[i]-gmem_need)/self.gmem[i]
                ave_per=(cpu_rest_per+mem_rest_per+gpu_rest_per+gmem_rest_per)/4
                satisfy_gpu_id_list.append([i,ave_per])
                satisfy_score+=ave_per
                
        satisfy_gpu_id_list.sorted(key=lambda x:x[1], reverse=True)
        return satisfy_gpu_id_list, satisfy_score
    
    
    def get_over_corss_num(self):
        if self.cross_gpu_job_num<self.max_cross_gpu_job_num:
            return 0
        else:
            return self.cross_gpu_job_num-self.max_cross_gpu_job_num+1
        
    def get_idle_port(self):
        while is_port_in_use(self.current_port):
            self.current_port+=1
        self.current_port+=1
        
        return self.current_port
        
        
    
        