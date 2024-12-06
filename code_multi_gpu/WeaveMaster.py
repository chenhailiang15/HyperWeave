from socket import  *
import time


master_ip="10.26.128.51"
master_socket_port=8000



# 1.创建套接字
tcp_socket = socket(AF_INET,SOCK_STREAM)
# 2.准备连接服务器，建立连接
tcp_socket.connect((master_ip,master_socket_port))  # 连接服务器，建立连接,参数是元组形式
#准备需要传送的数据
send_data = "我要开始发送数据啦！"
tcp_socket.send(send_data.encode()) 
while True:
    message=input("输入要发送的数据：")
    if message=="":
        break
    else:
        tcp_socket.send(message.encode()) 
# time.sleep(5000)


#从服务器接收数据
# while True:
#注意这个1024byte，大小根据需求自己设置
# from_server_msg = tcp_socket.recv(1024)
#加上.decode("gbk")可以解决乱码
# print(from_server_msg.decode("gbk"))  
#关闭连接
tcp_socket.close()