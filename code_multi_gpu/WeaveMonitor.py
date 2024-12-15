import threading
import numpy as np

class WeaveMonitor:
    def __init__(self,overshared_factor, print_level):
        self.print_level=print_level
        self.lock=threading.Lock()
        self.overshared_factor=overshared_factor
        self.cross_num=0
        self.cross_master_gpu_num=0
        self.cross_worker_gpu_num=0
    
    def set_master_resource(self,cpu, mem, gpu, gmem):
        self.master_cpu=cpu
        self.master_mem=mem
        self.master_gpu=np.array([100*self.overshared_factor for i in range(gpu)])
        self.master_gmem=np.array([gmem for i in range(gpu)])
        
    def set_worker_resource(self,cpu, mem, gpu, gmem):
        self.worker_cpu=cpu
        self.worker_mem=mem
        self.worker_gpu=np.array([100*self.overshared_factor for i in range(gpu)])
        self.worker_gmem=np.array([gmem for i in range(gpu)])
        
    def alloc_resource(self,node_kind, cpu, mem, gpu, gmem):
        with self.lock:
            if node_kind=="master":
                self.master_cpu-=cpu
                self.master_mem-=mem
                self.master_gpu=self.master_gpu-gpu
                self.master_gmem=self.master_gmem-gmem
            elif node_kind=="worker":
                self.worker_cpu-=cpu
                self.worker_mem-=mem
                self.worker_gpu=self.worker_gpu-gpu
                self.worker_gmem=self.worker_gmem-gmem
            else:
                print("node kind wrong!")
                exit(256)
    def takeback_resource(self,node_kind, cpu, mem, gpu, gmem):
        with self.lock:
            if node_kind=="master":
                self.master_cpu+=cpu
                self.master_mem+=mem
                self.master_gpu=self.master_gpu+gpu
                self.master_gmem=self.master_gmem+gmem
            elif node_kind=="worker":
                self.worker_cpu+=cpu
                self.worker_mem+=mem
                self.worker_gpu=self.worker_gpu+gpu
                self.worker_gmem=self.worker_gmem+gmem
            else:
                print("node kind wrong!")
                exit(256)
                
                
    def get_satisfy_gpu_id():
        
        
        
        
        return [[(gpuid, )],[]]
    
    
    
    
    
    
    
