from NodeCommunicate import CommunicateClient
import json
import threading
import os
from Job import Job
import time


class WeaveWorker:
    def __init__(self,print_level):
        self.print_level=print_level
        self.comunicator=CommunicateClient("10.26.128.115",print_level=print_level)
        self.comunicator.start_connect()
        self.comunicator.start_listening(self.message_receive)

        
    def message_receive(self,message):
        print("master receive:\n", message)
        job=Job()
        job.load_string(message)
        job.set_start_time(time.time())
        
        sub_thread=threading.Thread(target=self.run_command,args=(job, job.command,))
        sub_thread.start()
        
        
        
    def run_command(self,job, message):
        print("worker receive message:\n",message)
        back=os.system(message)
        print("命令执行返回结果：",back)


if __name__=="__main__":
    print_level=10
    WeaveWorker(print_level=print_level)