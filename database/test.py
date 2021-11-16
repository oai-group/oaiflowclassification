'''创建数据库'''
import pymysql
#打开数据库连接，不需要指定数据库，因为需要创建数据库
conn = pymysql.connect(host = 'localhost',user = "root",passwd = "123456")
#获取游标
cursor=conn.cursor()
#创建pythonBD数据库
cursor.execute('CREATE DATABASE IF NOT EXISTS pythonDB DEFAULT CHARSET utf8 COLLATE utf8_general_ci;')
cursor.close()#先关闭游标
conn.close()#再关闭数据库连接
print('创建pythonBD数据库成功')

