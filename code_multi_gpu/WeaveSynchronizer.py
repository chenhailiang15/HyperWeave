import threading
import time
from multiprocessing import shared_memory
import multiprocessing
import numpy as np

class Synchronizer:
    def __init__(self,prior,shm_name,shm_size=3*np.dtype(np.int8).itemsize,enable_flage=True):
        self.cpu_index=0
        self.gpu_index=1
        self.prior=prior
        self.shm_name=shm_name
        self.shm_size=shm_size
        self.enable_flage=enable_flage
        if self.enable_flage:
            
            if self.load_share_memory():#为True，表示创建新的，需要初始化
                self.boolean_array = np.ndarray((3,), dtype=np.int8, buffer=self.shm.buf)
                new_values = np.array([True, False,False], dtype=bool)
                self.boolean_array[:]=new_values.astype(np.int8)
            else:
                self.boolean_array = np.ndarray((3,), dtype=np.int8, buffer=self.shm.buf)
        return
    
    def __init__(self,shm_name,shm_size=4*np.dtype(np.int8).itemsize):
        self.shm_name=shm_name
        self.shm_size=shm_size
        
        if self.load_share_memory():#为True，表示创建新的，需要初始化
            self.boolean_array = np.ndarray((4,), dtype=np.int8, buffer=self.shm.buf)
            new_values = np.array([False, False,False,False], dtype=bool)
            self.boolean_array[:]=new_values.astype(np.int8)
        else:
            self.boolean_array = np.ndarray((4,), dtype=np.int8, buffer=self.shm.buf)
        return

    def load_share_memory(self):
        try:
            self.shm=shared_memory.SharedMemory(name=self.shm_name)
            print(f"共享内存 '{self.shm_name}' 已存在。")
            return False
        except FileNotFoundError:
            self.shm=shared_memory.SharedMemory(name=self.shm_name, create=True, size=self.shm_size)
            print(f"共享内存 '{self.shm_name}' 不存在，现在创建。")
            return True
        
        
            
    def sync_in_start_epoch(self,first_epoch):
        #是否发挥作用
        if not self.enable_flage:
            return
        #初始化为 CPU： True， GPU：False
        if first_epoch & self.prior:
            return
            
        while self.boolean_array[self.cpu_index] == True:
            time.sleep(0.1)
        self.boolean_array[self.cpu_index]=True
        return
        
    def sync_in_batch(self):
        #是否发挥作用
        if not self.enable_flage:
            return
        
        self.boolean_array[self.cpu_index]=False
        while self.boolean_array[self.gpu_index] == True:
            time.sleep(0.1)
        self.boolean_array[self.gpu_index]=True
            
        return
        
        
    def sync_in_end_epoch(self):
        #是否发挥作用
        if not self.enable_flage:
            return
        
        self.boolean_array[self.gpu_index]=False
        return
    
    
    def close_unlink(self):
        #是否发挥作用
        if not self.enable_flage:
            return
        
        if self.boolean_array[2] == False:
            self.boolean_array[2]=True
        else:
            try:
                self.shm.close()
                self.shm.unlink()
                print("shared memory delete here!")
                
            except :
                print("shared memory has deleted!")
    
    
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