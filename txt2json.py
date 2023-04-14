import json
import numpy as np
import os

#车道线数据(.txt)存放路径
lanes_dir = r'F:\三维重建(毫末公司实习）\data\haomo_4dlabel\lidar_label\lidar_lanes'

#生成的json文件保存路径
save_dir = r"F:\三维重建(毫末公司实习）\data\haomo_4dlabel\lidar_label\lidar_lanes"


lidar_lanes = {}
txt_names = os.listdir(lanes_dir)
for idx, txt_name in enumerate(txt_names):
    txt_file = os.path.join(lanes_dir, txt_name)
    lane = np.loadtxt(txt_file)
    lidar_lanes["lane{}".format(idx)] = []
    for i in range(lane.shape[0]):
        x, y, z = lane[i]
        lidar_lanes["lane{}".format(idx)].append({"x": x, "y": y, "z": z})
    print(lane)
json.dump(lidar_lanes, open(save_dir + "\lane_00005_lidar.json", "w"))






