import time
from NodeCommunicate import NodeMessageSender





if __name__=="__main__":
    worker_ip="10.26.128.51"
    worker_port=8000
    node_message_sender=NodeMessageSender(worker_ip,worker_port)
    while True:
        send_info=input("发送数据：")
        node_message_sender.send(send_info)
        
