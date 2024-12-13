from NodeCommunicate import CommunicateClient
import json
import threading
import os
from Job import Job
import time
import queue

class WeaveWorker:
    def __init__(self,print_level):
        self.print_level=print_level
        self.comunicator=CommunicateClient("10.26.128.115",print_level=print_level)
        self.comunicator.start_connect()
        self.comunicator.start_listening(self.message_receive)
        self.sub_thread_queue=queue.Queue()
        self.buffer=""
        
    def message_receive(self,message):
        self.buffer+=message
        buffer_list=self.buffer.split("--end")
        if len(buffer_list)>1:
            for i in range(len(buffer_list)-1):
                job=Job()
                job.load_string(buffer_list[i])
                job.set_start_time(time.time())
                
                sub_thread=threading.Thread(target=self.run_command,args=(job, job.command,))
                sub_thread.start()
                self.sub_thread_queue.put(sub_thread)
                
            self.buffer=buffer_list[len(buffer_list)-1]
        
        
        
        
    def run_command(self,job, message):
        print("worker receive message:\n",message)
        back=os.system(message)
        if back==0:
            job.succeed()
        else:
            job.failed()
            
        job.set_end_time(time.time())
        self.comunicator.send(job.to_string()+"--end")
            
        print("命令执行返回结果：",back)


if __name__=="__main__":
    print_level=10
    WeaveWorker(print_level=print_level)