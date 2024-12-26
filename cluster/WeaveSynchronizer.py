import threading
import time
from multiprocessing import shared_memory
import multiprocessing
import numpy as np

class Synchronizer:
    def __init__(self,shm_name,shm_size=3,prior=True,enable_flage=True,max_sync_num=0):
        self.enable_flage=enable_flage
        
        if not enable_flage:
            return    
            
        if shm_size==3:
            self.cpu_index=0
            self.gpu_index=1
            self.prior=prior
            self.shm_name=shm_name
            self.shm_size=shm_size
            self.enable_flage=enable_flage
            
            
            if self.enable_flage:
                
                if self.load_share_memory():#为True，表示创建新的，需要初始化
                    self.boolean_array = np.ndarray((3,), dtype=np.int8, buffer=self.shm.buf)
                    new_values = np.array([True, False, False], dtype=bool)
                    self.boolean_array[:]=new_values.astype(np.int8)
                else:
                    self.boolean_array = np.ndarray((3,), dtype=np.int8, buffer=self.shm.buf)
        
        elif shm_size==4:
            self.shm_name=shm_name
            self.shm_size=shm_size
            if self.load_share_memory():#为True，表示创建新的，需要初始化
                self.boolean_array = np.ndarray((4,), dtype=np.int8, buffer=self.shm.buf)
                new_values = np.array([False, False, False, False], dtype=bool)
                self.boolean_array[:]=new_values.astype(np.int8)
            else:
                self.boolean_array = np.ndarray((4,), dtype=np.int8, buffer=self.shm.buf)
        elif shm_size==12:
            
            self.shm_name=shm_name
            self.shm_size=shm_size
            if self.load_share_memory():#为True，表示创建新的，需要初始化
                self.boolean_array = np.ndarray((3,4), dtype=np.int8, buffer=self.shm.buf)
                if max_sync_num==4:
                    new_values = np.array([[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]], dtype=np.int8)
                elif max_sync_num==3:
                    new_values = np.array([[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 1]], dtype=np.int8)
                elif max_sync_num==2:
                    new_values = np.array([[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 1, 1]], dtype=np.int8)
                elif max_sync_num==1:
                    new_values = np.array([[0, 0, 0, 0],[0, 0, 0, 0],[0, 1, 1, 1]], dtype=np.int8)
                else:
                    print("max sync wrong!")
                    exit(-1)
                    
                self.boolean_array[:]=new_values.astype(np.int8)
            else:
                self.boolean_array = np.ndarray((3,4), dtype=np.int8, buffer=self.shm.buf)
                
        elif shm_size==16:
            self.shm_name=shm_name
            self.shm_size=shm_size
            if self.load_share_memory():#为True，表示创建新的，需要初始化
                self.boolean_array = np.ndarray((4,), dtype=np.float32, buffer=self.shm.buf)
                new_values = np.array([0,0,0,0], dtype=np.float32)
                self.boolean_array[:]=new_values.astype(np.float32)
            else:
                self.boolean_array = np.ndarray((4,), dtype=np.float32, buffer=self.shm.buf)
        else:
            print("share memory size wrong!")
            exit(-1)
        return

    def load_share_memory(self):
        try:
            self.shm=shared_memory.SharedMemory(name=self.shm_name)
            print(f"共享内存 '{self.shm_name}' 已存在。size:{self.shm_size}")
            return False
        except FileNotFoundError:
            self.shm=shared_memory.SharedMemory(name=self.shm_name, create=True, size=self.shm_size)
            print(f"共享内存 '{self.shm_name}' 不存在，现在创建。size:{self.shm_size}")
            return True
        
        
    
    def sync_in_start_epoch(self,first_epoch):
        #是否发挥作用
        if not self.enable_flage:
            return
        #初始化为 CPU： True， GPU：False
        if first_epoch & self.prior:
            return
            
        while self.boolean_array[self.cpu_index] == True:
            a=1
        self.boolean_array[self.cpu_index]=True
        return
        
    def sync_in_batch(self):
        #是否发挥作用
        if not self.enable_flage:
            return
        
        self.boolean_array[self.cpu_index]=False
        while self.boolean_array[self.gpu_index] == True:
            a=1
        self.boolean_array[self.gpu_index]=True
            
        return
        
        
    def sync_in_end_epoch(self):
        #是否发挥作用
        if not self.enable_flage:
            return
        
        self.boolean_array[self.gpu_index]=False
        return

    
    def muri_sync_start(self, idx_on_gpu, stage_id):
        #是否发挥作用
        if not self.enable_flage:
            return
        
        while True:
            if self.boolean_array[1,stage_id]==0:
                leave_to_job_id=self.boolean_array[0,stage_id]
                if idx_on_gpu==leave_to_job_id:
                    break
                if self.boolean_array[2,leave_to_job_id] == 1:
                    break
                
        self.boolean_array[1,stage_id]=1
        
        return
    
    def muri_sync_end(self, idx_on_gpu, stage_id):
        #是否发挥作用
        if not self.enable_flage:
            return
        next_job_idx=idx_on_gpu
        while True:
            next_job_idx=next_job_idx+1 if next_job_idx<3 else 0
            
            if self.boolean_array[2,next_job_idx] == 0:
                    break 
            
        
        self.boolean_array[0,stage_id]=next_job_idx
        self.boolean_array[1,stage_id]=0
        
    def muri_job_end(self, idx_on_gpu):
        self.boolean_array[2,idx_on_gpu]=1

        
        return
    
    def close_unlink(self):
        #是否发挥作用
        if not self.enable_flage:
            return
        
        if self.boolean_array[2] == False:
            self.boolean_array[2]=True
            # self.shm.unlink()
            print("shared memory close here!")
            
        else:
            self.shm.close()
            self.shm.unlink()
            # self.shm.unlink()
            # self.shm.close()
            print("shared memory delete here!")
            
    
    
    def set_value(self, index, value):
        self.boolean_array[index]=value
        
        
    def get_value(self, index):
        return self.boolean_array[index]
    
    def close_only(self):
        try:
            self.shm.close()
            print("shared memory close here!")
            
        except :
            print("shared memory has deleted!")
            
    def delete_shm(self):
        try:
            self.shm.close()
            self.shm.unlink()
            print("shared memory delete here!")
            
        except :
            print("shared memory has deleted!")
            
            
            
    def write_test(self):
        for i in range(10):
            
            self.boolean_array[i%2]= not self.boolean_array[i%2]
            print("write:",self.boolean_array.astype(bool))
            time.sleep(2)
            
            
    def read_test(self):
        for i in range(20):
            print("read:",self.boolean_array.astype(bool))
            time.sleep(1)
            
            
            







def threading_func(first_flage, shm_name,shm_size):
    sync_er=Synchronizer(first_flage, shm_name,shm_size)
    # if first_flage:
    #     sync_er.write_test()
    # else:
    #     sync_er.read_test()
    thread_id=threading.current_thread().ident.__str__()
        
    
    for i in range(10):
        sync_er.sync_in_start_epoch(i==0)
        print(thread_id+":"+"load data ... use cpu")
        time.sleep(0.1)
        for j in range(10):
            if j == 0:
                sync_er.sync_in_batch()
            print(thread_id+":"+"train model ... use gpu")
            time.sleep(0.01)
            
        sync_er.sync_in_end_epoch()
    
    print("code over")
    sync_er.close()
    print("close over")
            




if __name__=="__main__":
    
    shm_name="test_mem_share_sync"
    shm_size=shm_size = 3 * np.dtype(np.int8).itemsize
    subthreading1=threading.Thread(target=threading_func,args=(True,shm_name,shm_size))
    subthreading2=threading.Thread(target=threading_func,args=(False,shm_name,shm_size))
    subthreading1.start()
    subthreading2.start()
    
    subthreading1.join()
    subthreading2.join()
    
    print("process end!")