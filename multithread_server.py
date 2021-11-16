from socket import *
from threading import Thread
from pool import *
import json
import struct

 
class TcpServer(object):
    """Tcp服务器"""
    def __init__(self, Port):
        """初始化对象"""
        self.code_mode = "utf-8"    #收发数据编码/解码格式
        self.server_socket = socket(AF_INET, SOCK_STREAM)   #创建socket
        self.server_socket.setsockopt(SOL_SOCKET, SO_REUSEADDR, True)   #设置端口复用
        self.server_socket.bind((SEVER_HOST, Port))     #绑定IP和Port
        self.server_socket.listen(100)  #设置为被动socket
        print("服务器正在监听...")
 
    def run(self):
        """运行"""
        while True:
            # 判断程序是否结束
            """
            mutex.acquire()
            if len(STOP_CLASSIFY) == 1:
                print("程序结束")
                mutex.release()
                break
            mutex.release()
            """

            client_socket, client_addr = self.server_socket.accept()    #等待客户端连接
            print("{} 已上线".format(client_addr))
            #创建线程为客户端服务
            tr = Thread(target=self.recv_data, args=(client_socket, client_addr))
            # if !tr.is_alive():
            tr.start()  #开启线程
 
        self.server_socket.close()
 

    def recv_data(self, client_socket, client_addr):
        """收发数据"""
        while True:
            # 判断程序是否结束
            """
            mutex.acquire()
            if len(STOP_CLASSIFY) == 1:
                print("程序结束")
                mutex.release()
                break
            mutex.release()
            """

            
            recv = client_socket.recv(80)
            # a = []
            data = []
            if recv:
                # print('服务端收到客户端发来的消息:%s' % (recv))
                recv = recv[:77]
                a = struct.unpack('2i2f1i2l2d1l13B', recv)
                print(a)
                data = [[], []]
                idStr = ""
                mystr = a[10:]
                for i in range(10):
                    data[0].append(a[i]+0.0)
                for i in range(8):
                    if i % 4 == 3:
                        idStr += str(mystr[i])+"  "
                    else:
                        idStr += str(mystr[i]) + "."
                idStr += str(int(mystr[8])*256+int(mystr[9])) + "  "
                idStr += str(int(mystr[10])*256+int(mystr[11])) + "  "
                """
                if int(mystr[12]) == 6:
                    idStr += "TCP"
                else:
                    idStr += "UDP"
                """
                idStr += str(int(mystr[12])) +""
                data[1].append(idStr)

            


            #data = client_socket.recv(BUFFER_SIZE).decode(self.code_mode)
            #if len(data) > 8:
            #    data = json.loads(data)
            if data:
                # 关闭连接的客户端的线程
                if data == "exit":
                    # 发送关闭客户端的命令
                    client_socket.send("close connecting".encode(self.code_mode))
                    print("{} 已下线".format(client_addr))
                    break
                # 如果接收的命令是close 修改标志位 停止程序
                elif data == "close":
                    mutex.acquire()
                    close_program()
                    mutex.release()
                    break
                # 如果发送的命令是 send 则将分类结果发送过来
                elif data == "send":
                    mutex.acquire()
                    if CLASSIFY_RESULT:
                        for res in CLASSIFY_RESULT:
                            client_socket.send(str(res).encode(self.code_mode))
                        CLASSIFY_RESULT.clear()
                    else:
                        client_socket.send("暂无分类结果".encode(self.code_mode))
                    mutex.release()
                else:
                    # 将数据保存在PREDECT_FLOWS中 加锁

                    mutex.acquire()
                    if data not in PREDECT_FLOWS:
                        PREDECT_FLOWS.append(data)
                    mutex.release()
                    print("{}:发送特征的五元组为{}".format(client_addr, data[1]))
                    # print("缓冲池中有: ", len(PREDECT_FLOWS), " 个待分类的特征，请分类")
                    # client_socket.send(data.encode(self.code_mode))
            else: 	
                #客户端断开连接
                print("{} 已下线".format(client_addr))
                break

        client_socket.close()
 

def boot_server():
    # print("\033c", end="") 	#清屏
    # port = int(input("请输入要绑定的Port:"))
    my_server = TcpServer(SEVER_PORT)
    my_server.run()
 

if __name__ == "__main__":
    boot_server()
