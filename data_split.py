import glob
import shutil
import random
import os
import time


start_time = time.time()
root_dir = r"F:\BaiduNetdiskDownload\deepglobe-road-dataset\train\train"
save_dir = r'F:\deepglobal_road'


data_files = glob.glob(os.path.join(root_dir, "*_sat.jpg" ))
#设置随机种子
random.seed(42)
random.shuffle(data_files)

#划分比例
nums = len(data_files)
ratio = 0.8

#训练集
for i in range(int(nums * ratio)):
    img_dir = save_dir + r"\train\images"
    label_dir = save_dir + r'\train\labels'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    data_name = data_files[i].split("\\")[5]
    idx = data_name.split(".")[0].split("_")[0]
    shutil.copy(data_files[i], os.path.join(img_dir, data_name))
    shutil.copy(os.path.join(root_dir, idx + "_mask.png"), os.path.join(label_dir, idx + "_mask.png"))

#验证集
for i in range(int(nums * ratio), nums):
    img_dir = save_dir + r"\val\images"
    label_dir = save_dir + r'\val\labels'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    data_name = data_files[i].split("\\")[5]
    idx = data_name.split(".")[0].split("_")[0]
    shutil.copy(data_files[i], os.path.join(img_dir, data_name))
    shutil.copy(os.path.join(root_dir, idx + "_mask.png"), os.path.join(label_dir, idx + "_mask.png"))

end_time = time.time()
print("创建数据集共用时：{}min".format((end_time-start_time)/60))






