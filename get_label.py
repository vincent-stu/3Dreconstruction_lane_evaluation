import os

id_file = "63a405bde00bd020df909f7a.txt"
lujing_file = "63a6feb9caa724e1a94532ff.txt"

sequences_id = ["00005", "00014", "00015", "00017", "00021", "00031", "00042"]

with open(id_file, 'r') as f:
    ids = f.readlines()
flags = []
for id in sequences_id:
    temp = ids[int(id)].strip("\n")
    temp = temp.split("/")
    flag = temp[11][17:38]
    flags.append(flag)
#print(flags)


with open(lujing_file, 'r') as f:
    lujings = f.readlines()


label_paths = open("label_paths.txt", 'w')
for flag in flags:
    """依次处理每一个序列数据(每一个序列包含100张图)"""
    for lujing in lujings:
        if flag in lujing:
            #lujing = lujing.strip("\n")
            label_paths.writelines(lujing)
label_paths.close()



