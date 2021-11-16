"""
该文件用于将分类好的结果保存到mysql数据库中
"""


import pymysql
from pool import *


# 建立连接
def connect_database(addr, uer_name, usr_paswd, db_name):
	return pymysql.connect(host = addr, user = uer_name, passwd = usr_paswd, database = db_name, charset='utf8' )



# 插入数据
def mysqldb_insert(con, srcIP, dstIP, srcPort, dstPort, protocol,flowType):
    # con = connect_database(DB_ADDR, USER_NAME, USER_PASSWORD, DB_NAME)
    cursor = con.cursor()
    sql = "INSERT INTO measure(srcIP, dstIP, srcPort, dstPort, protocol, flowType) VALUES ({}, {}, {}, {}, {},\'{}\') ON DUPLICATE KEY UPDATE flowType = \'{}\'".format(srcIP, dstIP, srcPort, dstPort, protocol, flowType,flowType)
    # print(sql)
    try:
        # 执行sql语句
        cursor.execute(sql)
        # 提交到数据库执行
        con.commit()
    except:
        # 发生错误时回滚
        con.rollback()
    #cursor.close()
    #con.close()
    print("插入数据成功！")


# 查询数据
def mysqldb_research():
	return


# 清空数据
def mysqldb_clearall(con):
	cursor = con.cursor()
	sql = "TRUNCATE TABLE measure;"
	try:
		# 执行sql语句
		cursor.execute(sql)
		# 提交到数据库执行
		con.commit()
	except:
		# 发生错误时回滚
		con.rollback()

	print("表已清空！")





if __name__ == "__main__":
	# write_database()
	con = connect_database(DB_ADDR, USER_NAME, USER_PASSWORD, DB_NAME)
	
	mysqldb_insert(con, 3,4,1,1,'UDP',"iot")
	mysqldb_insert(con, 1,2,1,1,1,"video")
	mysqldb_insert(con, 1,3,3,1,1,"video")
	mysqldb_insert(con, 1,4,3,1,1,"video")
	
	# mysqldb_clearall(con)
