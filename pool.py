import threading


"""
该文件定义使用的常量 和 进程锁
"""


# 设置服务器的IP和端口
SEVER_HOST = "127.0.0.1"	# 发送到windows测试
SEVER_PORT = 12345


# 缓冲区大小
BUFFER_SIZE = 8192


# 共享的全局变量 保存解析的特征 用于客户端发送和服务器端读取的缓冲区
PREDECT_FLOWS = []


# 共享的全局变量 保存分类的结果
CLASSIFY_RESULT = []


# 是否停止分类的标志位 长度为1退出程序
STOP_CLASSIFY = []


# 定义修改全局变量的函数
def close_program():
	global STOP_CLASSIFY
	STOP_CLASSIFY.append(1)


# 创建全局的进程锁
mutex = threading.Lock()


# 数据库操作
# 数据库地址
DB_ADDR = 'localhost'
# 数据库用户名
USER_NAME = 'root'
# 数据库用户密码
USER_PASSWORD = '123456'
# 数据库名称
DB_NAME = 'mytestdb'


# 测试使用的数据
test = [[221.0, 1350.0, 640.0, 376.26798960315506, 543.0, 
21257877.349853516, 4793407917.022705, 1263437211.5135193, 
2039103758.0566826, 119541525.84075928],["sip sport dip dport protocol"]]

