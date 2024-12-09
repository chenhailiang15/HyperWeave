import threading
import time
from multiprocessing import shared_memory
import multiprocessing
import numpy as np






class Synchronizer:
    def __init__(self,first_flage,shm_name,shm_size):
        self.first_flage=first_flage
        self.shm_name=shm_name
        self.shm_size=shm_size
        self.load_share_memory()
        self.boolean_array = np.ndarray((2,), dtype=np.int8, buffer=self.shm.buf)
        new_values = np.array([True, False], dtype=bool)
        self.boolean_array[:]=new_values.astype(np.int8)
        
        return
    
    
    def load_share_memory(self):
        try:
            self.shm=shared_memory.SharedMemory(name=self.shm_name)
            print(f"共享内存 '{self.shm_name}' 已存在。")
        except FileNotFoundError:
            self.shm=shared_memory.SharedMemory(name=self.shm_name, create=True, size=self.shm_size)
            print(f"共享内存 '{self.shm_name}' 不存在，现在创建。")
        
        
            
    def sync_in_start_epoch(self,fisrt_epoch):
        if fisrt_epoch & self.first_flage:
            return
        # while 
        
        return
        
    def sync_in_batch(self):
        return
        
        
    def sync_in_end_epoch(self):
        return
    
    
    def close(self):
        try:
            self.shm.close()
            self.unlink()
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
    if first_flage:
        sync_er.write_test()
    else:
        sync_er.read_test()
        
        
    sync_er.close()
    # for i in range(10):
    #     sync_er.sync_in_start_epoch(i)
    #     time.sleep(2)
    #     for j in range(10):
    #         sync_er.sync_in_batch()
    #         time.sleep(5)
            
    #     sync_er.sync_in_end_epoch()
    #     time.sleep(2)
            




if __name__=="__main__":
    
    shm_name="test_mem_share_sync"
    shm_size=shm_size = 2 * np.dtype(np.int8).itemsize
    subthreading1=threading.Thread(target=threading_func,args=(True,shm_name,shm_size))
    subthreading2=threading.Thread(target=threading_func,args=(False,shm_name,shm_size))
    subthreading1.start()
    subthreading2.start()
    
    subthreading1.join()
    subthreading2.join()
    
    print("process end!")