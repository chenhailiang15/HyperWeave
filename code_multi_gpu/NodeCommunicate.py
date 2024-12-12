from socket import  *
import time

class NodeMessageReceiver:
    def __init__(self, port=8000):
        self.port=port
        return
    def start_listening(self, func):

        with socket(AF_INET, SOCK_STREAM) as s:
            s.bind(("", self.port))
            s.listen()
            print(f"listening...")
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    instruct = conn.recv(1024).decode("gbk")
                    if not instruct:
                        break
                    func(instruct)
                    back_info=input("返回信息：")
                    conn.send(back_info.encode("gbk"))



class NodeMessageSender:
    def __init__(self,aim_node_ip, aim_node_port=8000):
        self.ip=aim_node_ip
        self.port=aim_node_port
        # 1.创建套接字
        self.tcp_socket = socket(AF_INET,SOCK_STREAM)
        # 2.准备连接服务器，建立连接
        self.tcp_socket.connect((self.ip,self.port))  # 连接服务器，建立连接,参数是元组形式
    
    def send(self,message):
        #发送数据
        self.tcp_socket.send(message.encode("gbk")) 
        back_data=self.tcp_socket.recv(1024).decode("gbk")
        print("接收到消息：",back_data )
    
    def close(self):
        #关闭连接
        self.tcp_socket.close()