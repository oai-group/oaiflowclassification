"""
预测函数：随机森林

每次运行前，检查：
四个需要修改的地方，命名是否正确
最后的运行模式是否正确

做预测 
1. 直接使用之前的模型做预测 分类结果和五元组封装起来 
2. 做一个线程池 多线程接收发送来的特征信息
"""
from common_utils import *  # 修改了
from argparse import ArgumentParser
from collections import namedtuple
from typing import List, Dict
from datetime import datetime
from path_utils import get_prj_root
import numpy as np
# from sklearn.externals import joblib
import joblib
import pprint
import socket
import string
from sklearn.ensemble import RandomForestClassifier  # 训练模型

random.seed(datetime.now())
model_dir = os.path.join(get_prj_root(), "classify/models")  # 修改：模型model文件夹路径
predict_model_pkl = os.path.join(model_dir, "dt2.pkl")  # 修改：模型的版本，只用修改此处就行

Instance = namedtuple("Instance", ["features", "label"])  # 实例

dirs = {
    "video": "./tmp/dt/video",
    "iot": "./tmp/dt/iot",
    "voip": "./tmp/dt/voip",
    "AR": "./tmp/dt/AR",
}
instances_dir = os.path.join(get_prj_root(), "classify/instances")  # 修改：instances路径

# 通过元组测试
test_flow = (1, 2, 3, 4, 5, 6, 7, 8, 6, 10)
test_flow2 = [221.0, 1350.0, 640.0, 376.26798960315506, 543.0, 
                21257877.349853516, 4793407917.022705, 1263437211.5135193, 
                2039103758.0566826, 119541525.84075928]


def train_and_predict():
    iot = load_pkl(os.path.join(instances_dir, "iot.pkl"))  # 不同实例的pkl是不同特征的
    videos = load_pkl(os.path.join(instances_dir, "video.pkl"))
    voip = load_pkl(os.path.join(instances_dir, "voip.pkl"))
    AR = load_pkl(os.path.join(instances_dir, "AR.pkl"))
    for i in videos:
        assert i.label == 0
    for i in iot:
        assert i.label == 1
    for i in voip:
        assert i.label == 2
    for i in AR:
        assert i.label == 3

    # print(videos)

    debug("# iot instances {}".format(len(iot)))
    debug("# video instances {}".format(len(videos)))
    debug("# VOIP instances {}".format(len(voip)))
    debug("# AR instances {}".format(len(AR)))

    random.shuffle(voip)  # 打乱排序
    random.shuffle(iot)
    random.shuffle(videos)
    random.shuffle(AR)

    n_video_train = int(len(videos) * 0.7)
    n_video_test = len(videos) - n_video_train

    video_train = videos[:n_video_train]
    video_test = videos[n_video_train:]

    iot_train = iot[:n_video_train]
    iot_test = iot[len(iot) - len(video_test):]

    voip_train = voip[:n_video_train]
    voip_test = voip[len(voip) - len(video_test):]

    AR_train = AR[:n_video_train]
    AR_test = AR[len(AR) - len(video_test):]

    info("#video train {}".format(len(video_train)))
    info("#iot train {}".format(len(iot_train)))
    info("#voip train {}".format(len(voip_train)))
    info("#AR train {}".format(len(AR_train)))

    train = []
    train.extend(iot_train)
    train.extend(video_train)
    train.extend(voip_train)
    train.extend(AR_train)
    random.shuffle(train)

    train_x = [x.features for x in train]
    train_y = [x.label for x in train]

    # test 1:1
    test = []

    info("#video test {}".format(len(video_test)))
    info("#iot test {}".format(len(iot_test)))
    info("#voip test {}".format(len(voip_test)))
    info("#AR test {}".format(len(AR_test)))

    test.extend(video_test)
    test.extend(iot_test)
    test.extend(voip_test)
    test.extend(AR_test)
    random.shuffle(test)

    test_x = [t.features for t in test]
    test_y = [t.label for t in test]

    # 训练以及预测
    predict_model = RandomForestClassifier(oob_score=True)  # 引入训练方法
    predict_model.fit(train_x, train_y)  # 队训练数据进行拟合
    predicts = predict_model.predict(test_x)
    """
    dt = DT()
    dt.fit((train_x, train_y))
    predicts = dt.predict(test_x)
    """
    """
    # 打印预测的结果
    print(predicts)
    print("-------------------------------")
    """

    # 保存模型
    fn_name = os.path.join(model_dir, predict_model_pkl)
    joblib.dump(predict_model, predict_model_pkl)

    # 评价模型
    count = 0
    for idx in range(len(test_x)):
        if int(predicts[idx]) == int(test_y[idx]):
            count += 1
    # print(count / len(test_x))
    return count / len(test_x)


def classify_flows(mode: 'int', predict_dir):
    """
    该函数用于训练模型并且测试模型的准确度 或者 预测结果
    :param mode: 0--训练模型    1--预测和分类流并返回
    :param predict_dir: 待预测的流的目录下的pkl文件
    :return: 待分类的流的分类结果列表
    """
    # 判断是只训练模型 还是 只是预测结果
    if mode == 0:
        # 此时训练使用数据训练模型 并且 保存模型 评价模型
        times = 10
        sum_predict = 0
        for _ in range(times):
            res = train_and_predict()
            sum_predict = sum_predict + res
        print("模型准确率为:", sum_predict / times)
    else:
        # 使用传递的文件来预测结果并且返回
        predict = load_pkl(os.path.join(predict_dir, "predict2.pkl"))

        test = []
        info("#video test {}".format(len(predict)))

        test.extend(predict)
        # random.shuffle(test)

        test_x = [t.features for t in test]

        predict_model = joblib.load(predict_model_pkl)
        predict_result = predict_model.predict(test_x)
        res_list = identify_classification(predict_result)
        return res_list


def classify_flow_list(flow_list):
    """
    该方法用于分类为元组的流
    格式：[[1, 2, 3, 4, 5, 6, 7, 8, 6, 10], ["五元组"]]
    """
    test_x = [flow_list[0]]
    predict_model = joblib.load(predict_model_pkl)
    predict_result = predict_model.predict(test_x)
    # 定义结果的变量 res = ["五元组", "分类结果"]
    res = []
    res.append(flow_list[1][0])

    # 得到字符串的结果
    if predict_result == 0:
        res.append("videos")
    elif predict_result == 1:
        res.append("iot")
    elif predict_result == 2:
        res.append("voip")
    else:
        res.append("AR")
    return res

def change_result_to_integer_list(result_list):
    # 将这个格式 res = ["五元组", "分类结果"] 改为[srcIP, dstIP, srcPort, dstPort, protocol, flowType]
    integer_list = []
    string_list = result_list[0].split()
    # 添加源ip 转换为int的
    integer_list.append(ipToLong(string_list[0]))
    # 添加目的ip 转换为int的
    integer_list.append(ipToLong(string_list[1]))
    # 添加源端口 转换为int的
    integer_list.append(int(string_list[2]))
    # 添加目的端口 转换为int的
    integer_list.append(int(string_list[3]))
    # 添加协议类型
    #if string_list[4] == 'TCP':
    #    integer_list.append(1)
    #elif string_list[4] == 'UDP':
    #    integer_list.append(2)
    integer_list.append(int(string_list[4]))
    # 添加分类结果
    integer_list.append(result_list[1])
    
    return integer_list

#将字符串形式的ip地址转成整数类型。
def ipToLong(ip_str):
    #print map(int,ip_str.split('.'))
    ip_long = 0
    for index,value in enumerate(reversed([int(x) for x in ip_str.split('.')])):
        ip_long += value<<(8*index)
    return ip_long


def identify_classification(predict_result):
    """
    该函数将分类结果的标签转换为具体内容字符串的结果
    :param predict_result:标签分类结果
    :return: 字符串分类结果
    """
    res_list = []
    for label in predict_result:
        if label == 0:
            res_list.append("videos")
        elif label == 1:
            res_list.append("iot")
        elif label == 2:
            res_list.append("voip")
        elif label == 3:
            res_list.append("AR")
    return res_list


if __name__ == '__main__':

    """
    测试 格式转换
    # 训练模型
    # classify_flows(mode=0, path=instances_dir)

    # 预测结果
    predict_dir = os.path.join(get_prj_root(), "classify/predict")  # 修改：instances路径
    predict_result_list = classify_flows(mode=1, predict_dir=predict_dir)
    pprint.pprint(predict_result_list)
    """



    list1 = ['54.52.52.53  51.49.50.44  8242  12596  UDP', 'iot']
    res = change_result_to_integer_list(list1)

    print(res)
