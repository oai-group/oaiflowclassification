"""
预测函数：随机森林

每次运行前，检查：
四个需要修改的地方，命名是否正确
最后的运行模式是否正确
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


# save_pkl(predict_model_pkl, knn.model)
# knn.save_model(dt_model_pkl) #储存模型


if __name__ == '__main__':
    n = 10
    s = 0
    predict_sum = 0
    for i in range(n):
        s = train_and_predict()
        predict_sum = predict_sum + s
    print(predict_sum / n)
