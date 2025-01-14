from util import *
import numpy as np
import threading



class Node:
    def __init__(self, master, node_id, node_name, ip, net_card, overshared_factor, print_level):
        self.master=master
        self.node_id=node_id
        self.node_name=node_name
        self.overshared_factor=overshared_factor
        
        self.print_level=print_level
        self.cross_gpu_job_num=0
        self.current_port=2000
        self.ip=ip
        self.net_card=net_card
        self.lock=threading.Lock()
        
        self.dealing_instance={}
        self.dealing_instance_name={}
        self.max_instance_num_for_single_gpu=3   #couple 算一个
        
    def set_init_resouce(self, cpu, mem, gpu_num, gmem, spec_gpu_id=[]):
        self.cpu=cpu
        self.mem=mem
        self.cpu_rest=cpu
        self.mem_rest=mem
        if len(spec_gpu_id)==0:
            self.gpu_num=gpu_num
            self.gpu=np.array([100*self.overshared_factor for i in range(gpu_num)])
            self.gmem=np.array([gmem for i in range(gpu_num)])
            self.gpu_rest=np.array([100*self.overshared_factor for i in range(gpu_num)])
            self.gmem_rest=np.array([gmem for i in range(gpu_num)])
        else:
            self.gpu_num= max(spec_gpu_id)+1
            self.gpu=np.array([0 for i in range(self.gpu_num)])
            self.gmem=np.array([0 for i in range(self.gpu_num)])
            self.gpu_rest=np.array([0 for i in range(self.gpu_num)])
            self.gmem_rest=np.array([0 for i in range(self.gpu_num)])
            for gpu_id in spec_gpu_id:
                self.gpu[gpu_id]=100*self.overshared_factor
                self.gmem[gpu_id]=gmem
                self.gpu_rest[gpu_id]=100*self.overshared_factor
                self.gmem_rest[gpu_id]=gmem
        #初始化记录信息
        for index in range(self.gpu_num):
            self.dealing_instance[index]=[]
            self.dealing_instance_name[index]=[]
            
    def print_node_resource(self):
        print(f"@@@@@@node id: {self.node_id}\tcpu-{self.cpu_rest}\tmem-{self.mem_rest}",end="\t")
        for i in range(self.gpu_rest.shape[0]):
            print(f"gpu_id-{i}-{self.gpu_rest[i]}-{self.gmem_rest[i]}",end="\t")
        print("")
        
    def alloc_resource(self, cpu, mem, gpu, gmem, gpu_id_list):
        with self.lock:
            if self.print_level > 10:
                self.print_node_resource()
                print(f"      node id: {self.node_id} need resource cpu-{cpu}\tmem-{mem}\tgpu-{gpu}\tgmem-{gmem}\tgpu id list-{gpu_id_list}")
            self.cpu_rest-=cpu
            self.mem_rest-=mem
            
            for gpu_id in gpu_id_list:
                self.gpu_rest[gpu_id]=self.gpu_rest[gpu_id]-gpu
                self.gmem_rest[gpu_id]=self.gmem_rest[gpu_id]-gmem
            if self.print_level > 10:
                self.print_node_resource()
        
    def takeback_resource(self, cpu, mem, gpu, gmem, gpu_id_list):
        with self.lock:
            if self.print_level > 10:
                self.print_node_resource()
                print(f"      node id: {self.node_id} takeback resource cpu-{cpu} mem-{mem} gpu-{gpu} gmem-{gmem}, gpu id list-{gpu_id_list}")
            self.cpu_rest+=cpu
            self.mem_rest+=mem
            for gpu_id in gpu_id_list:
                self.gpu_rest[gpu_id]=self.gpu_rest[gpu_id]+gpu
                self.gmem_rest[gpu_id]=self.gmem_rest[gpu_id]+gmem
            if self.print_level > 10:
                self.print_node_resource()
            
            

    
    def get_satisfy_gpu_id(self, pack_resource):

        with self.lock:
            satisfy_gpu_id_list = []

            if pack_resource == None:  #返回空闲的GPU list
                for gpu_id in range(self.gpu_num):
                    if self.gpu_rest[gpu_id] == 100 * self.overshared_factor:
                        satisfy_gpu_id_list.append(gpu_id)

            else:
                cpu_need = pack_resource[0]
                mem_need = pack_resource[1]
                gpu_need = pack_resource[2]
                gmem_need = pack_resource[3]

                # assert gpu_need <= 100

                if self.cpu_rest < cpu_need or self.mem_rest < mem_need:
                    return satisfy_gpu_id_list, len(satisfy_gpu_id_list)
                cpu_rest_per = (self.cpu_rest - cpu_need) / self.cpu
                mem_rest_per = (self.mem_rest - mem_need) / self.mem

                for i in range(self.gpu_num):
                    if self.master.system=="Weave" and len(self.dealing_instance_name[i]) >= self.max_instance_num_for_single_gpu:
                        continue
                    
                    if self.gpu_rest[i] >= gpu_need and self.gmem_rest[i] >= gmem_need:
                        gpu_rest_per = (self.gpu_rest[i] - gpu_need) / self.gpu[i]
                        gmem_rest_per = (self.gmem_rest[i] - gmem_need) / self.gmem[i]
                        ave_per = (cpu_rest_per + mem_rest_per + gpu_rest_per + gmem_rest_per) / 4
                        satisfy_gpu_id_list.append([i, ave_per])

                if len(satisfy_gpu_id_list) > 0:
                    satisfy_gpu_id_list.sort(key=lambda x: x[1], reverse=True)  # 进行排序，降序
                    max_gpu_num=min(math.floor(self.cpu_rest/cpu_need), math.floor(self.mem_rest/mem_need))
                    satisfy_gpu_id_list=satisfy_gpu_id_list[:max_gpu_num]

            return satisfy_gpu_id_list, len(satisfy_gpu_id_list)
        
    def get_satisfy_gpu_num_by_cap(self,pack_resource):
        cpu_need = pack_resource[0]
        mem_need = pack_resource[1]
        gpu_need = pack_resource[2]
        gmem_need = pack_resource[3]

        if self.cpu < cpu_need or self.mem < mem_need or self.gpu_num==0:
            return 0
        if self.gmem[0]<gmem_need or self.gpu[0]<gpu_need:
            return 0

        satisfy_gpu_num = min(math.floor(self.cpu / cpu_need), math.floor(self.mem / mem_need), self.gpu_num)

        return satisfy_gpu_num
        
    def get_idle_port(self):
        with self.lock:
            while is_port_in_use(self.ip, self.current_port):
                self.current_port+=1
            idle_port=self.current_port
            self.current_port+=1
            
            return idle_port
        
        
    def get_idle_gpu_num(self):
        idel_gpu_num=0
        for gpu_id in range(self.gpu_num):
            if self.gpu_rest[gpu_id]==100*self.overshared_factor:
                idel_gpu_num+=1
        return idel_gpu_num
    
    def get_ave_allocate_resource(self):
        ave_cpu=(self.cpu-self.cpu_rest)/self.cpu
        ave_mem=(self.mem-self.mem_rest)/self.mem
        gpu_alloc=0
        gmem_alloc=0
        for i in range(self.gpu_num):
            gpu_alloc += self.gpu[i]-self.gpu_rest[i]
            gmem_alloc += self.gmem[i] - self.gmem_rest[i]

        ave_gpu=gpu_alloc/self.gpu_num/(self.gpu[0]/self.overshared_factor)
        ave_gmem=gmem_alloc/self.gpu_num/self.gmem[0]
        return [ave_cpu, ave_mem, ave_gpu, ave_gmem]
    
    
       # def get_over_corss_num(self):
    #     with self.lock:
    #         if self.cross_gpu_job_num<self.max_cross_gpu_job_num:
    #             return 0
    #         else:
    #             return self.cross_gpu_job_num-self.max_cross_gpu_job_num+1
    # def get_satisfy_gpu_id(self,pack_resource):
        
    #     with self.lock:
    #         satisfy_gpu_id_list=[]
    #         satisfy_score=0
    #         if pack_resource ==None:
    #             for gpu_id in range(self.gpu_num):
    #                 if self.gpu_rest[gpu_id]==100*self.overshared_factor:
    #                     satisfy_gpu_id_list.append(gpu_id)
    #                     satisfy_score+=1
    #         else:
    #             cpu_need=pack_resource[0]
    #             mem_need=pack_resource[1]
    #             gpu_need=pack_resource[2]
    #             gmem_need=pack_resource[3]
                
    #             if self.cpu_rest<cpu_need or self.mem_rest<mem_need:
    #                 return satisfy_gpu_id_list, satisfy_score
    #             cpu_rest_per=(self.cpu_rest-cpu_need)/self.cpu
    #             mem_rest_per=(self.mem_rest-mem_need)/self.mem
                
    #             for i in range(self.gpu_num):
    #                 if self.gpu_rest[i]>=gpu_need and self.gmem_rest[i]>=gmem_need:
    #                     gpu_rest_per=(self.gpu_rest[i]-gpu_need)/self.gpu[i]
    #                     gmem_rest_per=(self.gmem_rest[i]-gmem_need)/self.gmem[i]
    #                     ave_per=(cpu_rest_per+mem_rest_per+gpu_rest_per+gmem_rest_per)/4
    #                     satisfy_gpu_id_list.append([i,ave_per])
    #                     satisfy_score+=ave_per
    #             if len(satisfy_gpu_id_list)>0:
    #                 satisfy_gpu_id_list.sort(key=lambda x:x[1], reverse=True)  #进行排序，降序
    #         return satisfy_gpu_id_list, satisfy_score