from NodeCommunicate import NodeMessageReceiver
import json
import threading
import os




class WeaveWorker:
    def __init__(self):
        node_message_receiver=NodeMessageReceiver(8000)
        node_message_receiver.start_listening(self.do_action)

    def do_action(self, command):
        
        sub_thread=threading.Thread(target=self.run_command,args=(command,))
        sub_thread.start()
        
        
        
        
        
    def run_command(self,command):
        print("worker receive command:\n",command)
        os.system(command)

if __name__=="__main__":
    WeaveWorker()