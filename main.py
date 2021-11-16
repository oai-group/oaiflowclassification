from classify import *
from multithread_server import TcpServer, boot_server
from pool import *
from save_result import *
import json


def load_flows():
	# 一直执行
	con = connect_database(DB_ADDR, USER_NAME, USER_PASSWORD, DB_NAME) 	# 建立数据库的连接
	while True:
		# 判断是否停止识别
		"""
		mutex.acquire()
		if len(STOP_CLASSIFY) == 1:
			print("程序结束")
			mutex.release()
			break
		mutex.release()
		"""

		mutex.acquire()
		if PREDECT_FLOWS:
			# 读出所有的结果

			for feature in PREDECT_FLOWS:
				if type(feature) == list:
					# print("type: ", type(feature))
					# 预测结果
					predict_result = classify_flow_list(feature)
					
					# 将分类结果放到全局变量中
					# if predict_result not in CLASSIFY_RESULT:
					# 	CLASSIFY_RESULT.append(predict_result)
					#print("缓冲池中有: ", len(CLASSIFY_RESULT), "个已经分类好的结果，请取走")
					
					# 将分类结果转为整数
					integer_list = change_result_to_integer_list(predict_result)
					# 存入数据库
					mysqldb_insert(con, integer_list[0], integer_list[1], integer_list[2], integer_list[3], integer_list[4], integer_list[5])
					print(integer_list, "存入数据库成功！")
					print("分类结果为：", integer_list)
			# 清空列表
			PREDECT_FLOWS.clear()
		mutex.release()


# 定义一个函数作为主线程 让其他的守护他 判断关键字close后关闭主线称
def main_threading():
	global STOP_CLASSIFY
	# 创建线程
	t_server = threading.Thread(target = boot_server)
	t_classify = threading.Thread(target = load_flows)
	# 设置其他的为守护线程
	t_server.setDaemon(True)
	t_classify.setDaemon(True)
	# 启动线程
	t_server.start()
	t_classify.start()
	while True:
		mutex.acquire()
		if len(STOP_CLASSIFY) == 1:
			print("程序结束")
			mutex.release()
			break
		mutex.release()


if __name__ == "__main__":
	main_threading()



