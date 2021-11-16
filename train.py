"""
10个特征的（前五个包大小，后五个包间隔）：最小值，最大值，平均值，方差，中位数
每次运行前，检查：
四个需要修改的地方，命名是否正确
最后的运行模式是否正确

这个文件用于把pcap解析的文件生成特征和标签的形式 并且5个包一组
"""
from common_utils import *  # 修改了
from argparse import ArgumentParser
from collections import namedtuple
from typing import List, Dict
from path_utils import get_prj_root
from model import DT  # 修改了
from datetime import datetime
from path_utils import get_prj_root
import numpy as np
from sklearn.decomposition import PCA
import time

# start counting time
start = time.time()

random.seed(datetime.now())

# get the path of the models
model_dir = os.path.join(get_prj_root(), "classify/models")  # 修改：模型models路径
dt_model_pkl = os.path.join(model_dir, "dt2_9.pkl")  # 修改：模型的版本，只用修改此处就行

Instance = namedtuple("Instance", ["features", "label"])  # 实例

win_size = 5  # 窗口大小
limit = 100000

# the path
# iot-物联网流 video-视频流 voip-音频流 AR-AR流用的高清视频流代替
dirs = {
    "video": "./tmp/dt/video",
    "iot": "./tmp/dt/iot",
    "voip": "./tmp/dt/voip",
    "AR": "./tmp/dt/AR",
}
instances_dir = os.path.join(get_prj_root(), "classify/instances")  # 修改：instances路径


# 获取特征
def get_median(data):  # 产生中位数
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half]) / 2


def gen_single_instance(dir_name, flow, flow_type):
    # debug("generate {}".format(flow["file"]))
    def extract_features(raw_features: List[float]):  # 修改特征
        extracted_features = []
        raw_features = [r for r in raw_features if int(r) >= 0]

        extracted_features.append(min(raw_features))
        extracted_features.append(max(raw_features))
        extracted_features.append(sum(raw_features) / len(raw_features))
        extracted_features.append(np.std(raw_features))  # 标准差
        extracted_features.append(get_median(raw_features))  # 中位数
        return extracted_features

    features = []
    idts = []
    ps = []
    idt_file = os.path.join(dir_name, flow["idt"])  # 包大小
    ps_file = os.path.join(dir_name, flow["ps"])  # 包间隔
    with open(idt_file, 'r') as fp:
        lines = fp.readlines()
        fp.close()
    lines = [l.strip() for l in lines]  # .strip()用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。

    lines = [l for l in lines if len(l) > -1]
    if len(lines) > win_size:
        lines = lines[:win_size]
    for l in lines:
        idts.append(float(l))

    with open(ps_file, "r") as fp:
        lines = fp.readlines()
        fp.close()

    lines = [l.strip() for l in lines]
    lines = [l for l in lines if len(l) > 0]
    if len(lines) > win_size:
        lines = lines[:win_size]

    for l in lines:
        ps.append(float(l))

    # 有很奇怪的现象
    ps = [p for p in ps if p > 0]
    if len(ps) == 0:
        print(flow["ps"])
        return None
    idts = [i for i in idts if i >= 0]
    if len(idts) == 0:
        return None

    features.extend(extract_features(ps))  # 包间隔的数理统计
    features.extend(extract_features(idts))  # 包大小的数理统计
    if flow_type == "video":
        label = 0
    elif flow_type == "iot":
        label = 1
    elif flow_type == "voip":
        label = 2
    elif flow_type == "AR":
        label = 3
    else:
        err("Unsupported flow type")
        raise Exception("Unsupported flow type")
    return Instance(features=features, label=label)


def generate():
    instances_dir = os.path.join(get_prj_root(), "classify/instances")  # 修改：instances_dir实例的路径
    for flow_type, dirname in dirs.items():
        stats_fn = os.path.join(dirname, "statistics.json")  # statistics.json流量统计的文件
        debug(stats_fn)
        statistics = load_json(os.path.join(dirname, "statistics.json"))
        debug("#flows {}".format(statistics["count"]))
        flows: List = statistics["flows"]
        sorted(flows, key=lambda f: -f["num_pkt"])
        if len(flows) > limit:
            flows = flows[:limit]
        instances = [gen_single_instance(dirname, f, flow_type) for f in flows]
        instances = [i for i in instances if i is not None]
        debug("#{} instances {}".format(flow_type, len(instances)))
        # print(len(instances))
        save_pkl(os.path.join(instances_dir, "{}.pkl".format(flow_type)), instances)  # 保存Python内存数据到文件


if __name__ == '__main__':
    parser = ArgumentParser()
    print("running mode\n"
          "1. generate instances\n"
          "2. train dt\n")
    parser.add_argument("--mode", type=int, default=1)  # default为模式修改
    args = parser.parse_args()
    mode = int(args.mode)
    if mode == 1:
        generate()
    end = time.time()
    print("程序运行时间:%.2f秒" % (end - start))
