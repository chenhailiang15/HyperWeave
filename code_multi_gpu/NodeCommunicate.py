from socket import  *
import time
import threading





class CommunicateServer:
    def __init__(self, port=8000, print_level=0):
        self.port=port
        self.print_level=print_level
        return
    def start_listening(self, func):

        self.sock= socket(AF_INET, SOCK_STREAM)
        self.sock.bind(("", self.port))
        self.sock.listen()
        if self.print_level>0:
            print(f"listening...")
            
        self.conn, self.addr = self.sock.accept()
        
        if self.print_level>0:
            print(f"Connected by {self.addr}")
        while True:
            message = self.conn.recv(1024).decode("gbk")
            if not message:
                break
            func(message)
                    # back_info=input("返回信息：")
                    # conn.send(back_info.encode("gbk"))

    def send(self,message):
        self.conn.send(message.encode("gbk"))
        
        
    def close(self):
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()
        

class CommunicateClient:
    def __init__(self,aim_node_ip, aim_node_port=8000,print_level=0):
        self.ip=aim_node_ip
        self.port=aim_node_port
        # 1.创建套接字
        self.tcp_socket = socket(AF_INET,SOCK_STREAM)
        # 2.准备连接服务器，建立连接
        self.tcp_socket.connect((self.ip,self.port))  # 连接服务器，建立连接,参数是元组形式
        if print_level >0:
            print(f"connet succeed: ip-{aim_node_ip}, port-{aim_node_port}")

        
        
        
    def send(self,message):
        #发送数据
        self.tcp_socket.send(message.encode("gbk")) 
        # back_data=self.tcp_socket.recv(1024).decode("gbk")
        # print("接收到消息：",back_data )
    
    def start_listening(self, func):
        sub_thread=threading.Thread(target=self.__listening,args=(func,))
        sub_thread.start()
    
    
    def __listening(self,func):
        while True:
            back_data=self.tcp_socket.recv(1024).decode("gbk")
            func(back_data)
    
    def close(self):
        #关闭连接
        self.sock.shutdown(socket.SHUT_RDWR)
        self.tcp_socket.close()
        
        
    
    
    
    
    
    
def deal_info(received_info):
    print(received_info)
    
    
def send_info(sender):
    times=0
    while True:
        times+=1
        info=input("please input info:")
        if info=="break":
            break
        sender.send(info.encode("gbk"))
        if times>10:
            sender.close()
            break
        
    
    
     
if __name__=="__main__":
    is_server=True
    if is_server:
        c_server=CommunicateServer()
        c_server.start_listening(deal_info)
        send_info(c_server)
        
    else:
        c_client=CommunicateClient("10.26.128.115")
        c_client.start_listening(deal_info)
        send_info(c_client)