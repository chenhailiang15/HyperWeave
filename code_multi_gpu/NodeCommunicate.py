from socket import  *
import time
import threading


class CommunicateServer:
    def __init__(self, port=8000, print_level=0):
        self.port=port
        self.print_level=print_level
        return
    
    def start_connect(self):
        self.socket= socket(AF_INET, SOCK_STREAM)
        self.socket.bind(("", self.port))
        self.socket.listen()
        if self.print_level>0:
            print(f"waiting connect...")
            
        self.server_socket, self.addr = self.socket.accept()
        self.connect_flage=True
        if self.print_level>0:
            print(f"Connected by {self.addr}")

    def start_listening(self, func):
        sub_thread=threading.Thread(target=self.__listening,args=(func,))
        sub_thread.start()

    def __listening(self,func):
        while self.connect_flage:
            message=self.server_socket.recv(1024).decode("gbk")
            if message=="socket_close"  or not message:
                print("关闭socket")
                self.in_close()
                self.connect_flage=False
                break
            func(message)


    def send(self,message):
        if self.connect_flage:
            self.server_socket.send(message.encode("gbk"))
        
        
    def close(self):
        time.sleep(0.1)
        self.send("socket_close")
        self.connect_flage=False

    def in_close(self):
        self.server_socket.shutdown(SHUT_RDWR)
        self.server_socket.close()
        

class CommunicateClient:
    def __init__(self,ip, port=8000,print_level=0):
        self.ip=ip
        self.port=port
        self.print_level=print_level

    def start_connect(self):
        # 1.创建套接字
        self.client_socket = socket(AF_INET,SOCK_STREAM)
        # 2.准备连接服务器，建立连接
        self.client_socket.connect((self.ip,self.port))  # 连接服务器，建立连接,参数是元组形式
        self.connect_flage=True
        if self.print_level >0:
            print(f"connet succeed: ip-{self.ip}, port-{self.port}")
    
    def start_listening(self, func):
        sub_thread=threading.Thread(target=self.__listening,args=(func,))
        sub_thread.start()
    
    
    def __listening(self,func):
        while self.connect_flage:
            message=self.client_socket.recv(1024).decode("gbk")
            if message=="socket_close" or not message:
                print("关闭 socket")
                self.in_close()
                self.connect_flage=False
                break
            func(message)
    
    def send(self,message):
        if self.connect_flage:
            #发送数据
            self.client_socket.send(message.encode("gbk")) 
        # back_data=self.tcp_socket.recv(1024).decode("gbk")
        # print("接收到消息：",back_data )


    def close(self):
        #关闭连接
        self.send("socket_close")
        self.connect_flage=False

    def in_close(self):
        self.client_socket.shutdown(SHUT_RDWR)
        self.client_socket.close()
        
        
    
    
    
    
    
    
def deal_info(received_info):
    print(received_info)
    
    
def send_info(sender):
    times=0
    while True:
        times+=1
        info=input("\t\tplease input info:")
        if info=="break":
            break
        sender.send(info)
        if times>=5:
            sender.close()
            break
        
    
    
     
if __name__=="__main__":
    print_level=10
    is_server=True
    if is_server:
        c_server=CommunicateServer(print_level=print_level)
        c_server.start_connect()
        c_server.start_listening(deal_info)
        send_info(c_server)
        
    else:
        c_client=CommunicateClient("10.26.128.115",print_level=print_level)
        c_client.start_connect()
        c_client.start_listening(deal_info)
        send_info(c_client)