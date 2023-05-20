import os
import glob

dir_root = "/root/vincent/data/recon_label/"
txt_path = os.path.join(dir_root, "label_paths.txt")

jsonfile = glob.glob(os.path.join(dir_root, "*.json"))
print(jsonfile)

with open(txt_path, 'w') as f:
    for path in jsonfile:
        path = path + "\n"
        f.write(path)



