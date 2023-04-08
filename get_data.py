import shutil
import os

origin_path = "/root/vincent/ve_share2/fengyuan/haomo_4dlabel_v2.hds"
target_dir = "/root/vincent/data/haomo_4dlabel/"

# 读取存有数据路径的文件
with open(origin_path, 'r') as f:
    data_path = f.readlines()

data_path.sort()
# 要挑选的片段数量nums
nums = 50


for path in data_path:

    path = path.strip("\n")
    path = "/" + path
    dir_name = path.split(".")[0]
    exten_name = path.split(".")[1]

    group_name = dir_name.split("/")[8]    
    sequence_name = dir_name.split("/")[9]

    
    if int(sequence_name) <=nums:
        
        # 序列信息
        if group_name == "sequences":

            print("dir_name: ", dir_name)
            print("group_name: ", group_name)
            print("sequence_name:  ", sequence_name)
            class_name = dir_name.split("/")[10]
            print("class_name:  ", class_name)

            if class_name == "calib":
                save_dir = os.path.join(target_dir + group_name + "/", sequence_name + "/")
                print("save_dir: ", save_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                print("lujing:  ", os.path.join(save_dir, class_name + "." + exten_name))
                shutil.copy(path, os.path.join(save_dir, class_name + "." + exten_name))
            
            else:
                idx_name = dir_name.split("/")[11]
                save_dir = os.path.join(target_dir + group_name + "/", sequence_name + "/" + class_name + "/")
                print("save_dir: ", save_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                print("lujing:  ", os.path.join(save_dir, idx_name + "." + exten_name))
                shutil.copy(path, os.path.join(save_dir, idx_name + "." + exten_name))

        # 姿态信息
        if group_name == "poses":
            print("dir_name: ", dir_name)
            print("group_name: ", group_name)
            print("sequence_name:  ", sequence_name)
            save_dir = os.path.join(target_dir, group_name + "/")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print("lujing: ", os.path.join(save_dir, sequence_name + "." + exten_name))
            shutil.copy(path, os.path.join(save_dir, sequence_name + "." + exten_name))
        

            
