from NodeCommunicate import CommunicateClient
import json
import threading
import os
from Job import Job
import time
import queue

class WeaveWorker:
    def __init__(self,print_level):
        self.master_ip="10.26.128.51"
        self.master_port=8888
        self.print_level=print_level
        
        self.comunicator=CommunicateClient(ip=self.master_ip,port=self.master_port,print_level=print_level)
        self.comunicator.start_connect()
        sub_thread=self.comunicator.start_listening(self.message_receive)
        self.sub_thread_queue=queue.Queue()
        self.buffer=""
        sub_thread.join()
        
    def message_receive(self,message):
        self.buffer+=message
        buffer_list=self.buffer.split("--end")
        if len(buffer_list)>1:
            for i in range(len(buffer_list)-1):
                job=Job()
                job.load_string(buffer_list[i])
                
                sub_thread=threading.Thread(target=self.run_command,args=(job, job.command,))
                sub_thread.start()
                self.sub_thread_queue.put(sub_thread)
                
            self.buffer=buffer_list[len(buffer_list)-1]
        
        
        
        
    def run_command(self,job, message):
        print(f"******worker start job ${job.job_name}$ with command:\t",message)
        job.set_start_time(time.time())
        back=os.system(message)
        if back==0:
            job.succeed()
        else:
            job.failed()
            
        job.set_end_time(time.time())
        self.comunicator.send(job.to_string()+"--end")
            
        print(f"******worker end job ${job.job_name}$ with back code: {back}")
        
    def sub_thread_join(self):
        while self.sub_thread_queue.qsize()>0:
            sub_thread=self.sub_thread_queue.get()
            sub_thread.join()
        return
        
        

if __name__=="__main__":
    print_level=10
    weave_worker=WeaveWorker(print_level=print_level)
    weave_worker.sub_thread_join()
    print("The whole process end (by worker)!")