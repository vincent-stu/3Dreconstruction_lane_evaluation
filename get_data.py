import shutil
import os

origin_path = "/root/vincent/ve_share2/fengyuan/haomo_4dlabel_v2.hds"
target_dir = "/root/vincent/data/sequences/"

# 读取存有数据路径的文件
with open(origin_path, 'r') as f:
    data_path = f.readlines()

# 要挑选的片段数量nums
nums = 50
# 计数变量
count = 0

while count < nums:
    for path in data_path:
        path = path.strip("\n")
        path = "/" + path
        dir_name = path.split(".")[0]
        # print(dir_name)
        if len(dir_name) != 109:
            continue
        if int(dir_name[-21:-16]) <= 50:
            save_dir = os.path.join(target_dir, dir_name[-21:-6])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            shutil.copy(path, os.path.join(save_dir, dir_name[-6:]+".txt"))
            count += 1


