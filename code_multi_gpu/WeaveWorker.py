from NodeCommunicate import CommunicateClient
import json
import threading
import os




class WeaveWorker:
    def __init__(self,print_level):
        self.print_level=print_level
        self.comunicator=CommunicateClient("10.26.128.115")
        self.comunicator.start_connect()
        self.comunicator.start_listening(self.message_receive)

        
    def message_receive(self,message):
        print("master receive:\n", message)
        sub_thread=threading.Thread(target=self.run_command,args=(message,))
        sub_thread.start()
        
        
    def run_command(self,message):
        print("worker receive message:\n",message)
        # os.system(command)

if __name__=="__main__":
    print_level=10
    WeaveWorker(print_level=print_level)