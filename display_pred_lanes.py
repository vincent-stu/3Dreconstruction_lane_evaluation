import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json

root_dir = r'F:\reconstruction\recon_haomo_finetune_seq00_seq70'
pred_json_dir = r'F:\reconstruction\recon_haomo_finetune_seq00_seq70\pred_lane_json'
biaozhu_json_dir = r'F:\reconstruction\recon_haomo_finetune_seq00_seq70\biaozhu_lane_json'
E_dir = r'F:\reconstruction\recon_haomo_finetune_seq00_seq70\E'
I_dir = r'F:\reconstruction\recon_haomo_finetune_seq00_seq70\I'
rgb_dir = r'F:\reconstruction\recon_haomo_finetune_seq00_seq70\rgb'
save_dir = r'F:\reconstruction\recon_haomo_finetune_seq00_seq70\pred_lanes'
lanes_ids = ["lane0", "lane1"]

#sequences_ids = ["00000", "00005", "00006", "00014", "00015", "00017", "00031", "00037", "00044" , "00048", "00061"]
sequences_ids = ["00037"]
for sequences_id in sequences_ids:

    pred_json_name = "lane_ds_" + sequences_id + ".json"
    biaozhu_json_name = "lane_" + sequences_id + "_biaozhu.json"

    pred_lanes_json = os.path.join(pred_json_dir, pred_json_name)
    biaozhu_lanes_json = os.path.join(biaozhu_json_dir, biaozhu_json_name)
    #print("biaozhu_lanes_json: ", biaozhu_lanes_json)
    #print("pred_lanes_json: ", pred_lanes_json)
    pred_lanes = json.load(open(pred_lanes_json, 'r'))
    biaozhu_lanes  = json.load(open(biaozhu_lanes_json, 'r'))

    #重建点云上标注的车道线（自己标注的结果）
    pred_pts_world = dict()
    for key in pred_lanes.keys():
        pred_pts_world[key] = []
        for points in pred_lanes[key]:
            x = points["x"]
            y = points['y']
            z = points['z']
            a = 1.0
            pred_pts_world[key].append([x, y, z, a])
        pred_pts_world[key] = np.array(pred_pts_world[key]).reshape(-1, 4)
    #print("pred_pts_world: ", pred_pts_world)
    #print("len of pred_pts_world: ", len(pred_pts_world))
    #print("nums of pred_pts_world[lane0]: ", len(pred_pts_world["lane0"]))
    #print("nums of pred_pts_world[lane1]: ", len(pred_pts_world["lane1"]))

    # 雷达点云上标注的车道线（同事们标注的结果）
    biaozhu_pts_world = dict()
    for key in biaozhu_lanes.keys():
        biaozhu_pts_world[key] = []
        for points in biaozhu_lanes[key]:
            x = points["x"]
            y = points['y']
            z = points['z']
            a = 1.0
            biaozhu_pts_world[key].append([x, y, z, a])
        biaozhu_pts_world[key] = np.array(biaozhu_pts_world[key]).reshape(-1, 4)
    #print("biaozhu_pts_world: ", biaozhu_pts_world)
    #print("len of biaozhu_pts_world: ", len(biaozhu_pts_world))
    #print("nums of biaozhu_pts_world[lane0]: ", len(biaozhu_pts_world["lane0"]))
    #print("nums of biaozhu_pts_world[lane1]: ", len(biaozhu_pts_world["lane1"]))


    for i in range(100):
        id = str(i).zfill(6)
        save_dir_frame = os.path.join(save_dir,sequences_id)
        #save_dir_frame = os.path.join(save_dir_frame_temp, id)
        if not os.path.exists(save_dir_frame):
            os.makedirs(save_dir_frame)
        E_temp = os.path.join(E_dir, sequences_id)
        E_file = os.path.join(E_temp, id + '.npy')
        #print('path of E_file: ', E_file)
        I_temp = os.path.join(I_dir, sequences_id)
        I_file = os.path.join(I_temp, id + '.npy')
        #print("path of I_file: ", I_file)

        if not os.path.exists(E_file):
            continue

        E = np.load(E_file)
        I = np.load(I_file)
        rgb_temp = os.path.join(rgb_dir, sequences_id)
        rgb_path = os.path.join(rgb_temp, id + ".png" )
        #print("rgb_path: ", rgb_path)
        img = cv2.imread(rgb_path)
        #img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #print("shape of img: ", img.shape)
        #h, w, c = img.shape

        pred_pts_cam = dict()
        pred_pts_pixel = dict()
        pred_pts_uv = dict()
        for lanes_id in lanes_ids:
            # 依次将每一条车道线的三维空间坐标转换为二维图像坐标
            pred_pts_cam[lanes_id] = np.linalg.inv(E) @ pred_pts_world[lanes_id].T
            #print("shape of pred_pts_world[{}]".format(lanes_id), pred_pts_cam[lanes_id].shape)
            pred_pts_cam[lanes_id] = pred_pts_cam[lanes_id][:3]
            #print("pred_pts_cam[{}]: ".format(lanes_id), pred_pts_cam[lanes_id])
            #print("shape of pred_pts_cam[{}]: ".format(lanes_id), pred_pts_cam[lanes_id].shape)

            pred_pts_pixel[lanes_id] = (I[:3, :3] @ pred_pts_cam[lanes_id][:3]).T
            #print("pred_pts_pixel[{}]: ".format(lanes_id), pred_pts_pixel[lanes_id])
            #print("shape of p_pixel[{}]: ".format(lanes_id), pred_pts_pixel[lanes_id].shape)
            pred_pts_pixel[lanes_id] = pred_pts_pixel[lanes_id][:, :2] / np.expand_dims(pred_pts_pixel[lanes_id][:, 2], axis=1)
            #print("pred_pts_pixel[{}]: ".format(lanes_id), pred_pts_pixel[lanes_id])
            #print("shape of pred_pts_pixel[{}]: ".format(lanes_id), pred_pts_pixel[lanes_id].shape)
            #uv = np.round(p_pixel).astype(int)[0]
            pred_pts_uv[lanes_id] = np.round(pred_pts_pixel[lanes_id]).astype(int)

            #print("pred_pts_uv[{}]:  ".format(lanes_id), pred_pts_uv[lanes_id])
            #print("type of pred_pts_uv[{}]: ".format(lanes_id), type(pred_pts_uv[lanes_id]))
            #print("shape of pred_pts_uv[{}]: ".format(lanes_id), pred_pts_uv[lanes_id].shape)
            print("\n")
        print("pred_pts_cam: ", pred_pts_cam)
        print("pred_pts_uv: ", pred_pts_uv)
        print("shape of pred_pts_uv[lane0]: ", pred_pts_uv["lane0"].shape)
        print("shape of pred_pts_uv[lane1]: ", pred_pts_uv["lane1"].shape)


        biaozhu_pts_cam = dict()
        biaozhu_pts_pixel = dict()
        biaozhu_pts_uv = dict()
        for lanes_id in lanes_ids:
            # 依次将每一条车道线的三维空间坐标转换为二维图像坐标
            biaozhu_pts_cam[lanes_id] = np.linalg.inv(E) @ biaozhu_pts_world[lanes_id].T
            # print("shape of pred_pts_world[{}]".format(lanes_id), pred_pts_cam[lanes_id].shape)
            biaozhu_pts_cam[lanes_id] = biaozhu_pts_cam[lanes_id][:3]
            # print("pred_pts_cam[{}]: ".format(lanes_id), pred_pts_cam[lanes_id])
            # print("shape of pred_pts_cam[{}]: ".format(lanes_id), pred_pts_cam[lanes_id].shape)

            biaozhu_pts_pixel[lanes_id] = (I[:3, :3] @ biaozhu_pts_cam[lanes_id][:3]).T
            # print("pred_pts_pixel[{}]: ".format(lanes_id), pred_pts_pixel[lanes_id])
            # print("shape of p_pixel[{}]: ".format(lanes_id), pred_pts_pixel[lanes_id].shape)
            biaozhu_pts_pixel[lanes_id] = biaozhu_pts_pixel[lanes_id][:, :2] / np.expand_dims(biaozhu_pts_pixel[lanes_id][:, 2],
                                                                                        axis=1)
            # print("pred_pts_pixel[{}]: ".format(lanes_id), pred_pts_pixel[lanes_id])
            # print("shape of pred_pts_pixel[{}]: ".format(lanes_id), pred_pts_pixel[lanes_id].shape)
            # uv = np.round(p_pixel).astype(int)[0]
            biaozhu_pts_uv[lanes_id] = np.round(biaozhu_pts_pixel[lanes_id]).astype(int)

            # print("pred_pts_uv[{}]:  ".format(lanes_id), pred_pts_uv[lanes_id])
            # print("type of pred_pts_uv[{}]: ".format(lanes_id), type(pred_pts_uv[lanes_id]))
            # print("shape of pred_pts_uv[{}]: ".format(lanes_id), pred_pts_uv[lanes_id].shape)
            #print("\n")
        #print("biaozhu_pts_cam: ", biaozhu_pts_cam)
        #print("biaozhu_pts_uv: ", biaozhu_pts_uv)
        #print("shape of biaozhu_pts_uv[lane0]: ", biaozhu_pts_uv["lane0"].shape)
        #print("shape of biaozhu_pts_uv[lane1]: ", biaozhu_pts_uv["lane1"].shape)


        pred_pts_uv_json = dict()
        pred_pts_uv_json_save = dict()  # 用于保存json文件
        biaozhu_pts_uv_json = dict()
        biaozhu_pts_uv_json_save = dict()    # 用于保存json文件
        for lanes_id in lanes_ids:
            #在每一帧图像中，依次处理每一条车道线中，不属于该帧图像的点。
            pred_pts_uv_json[lanes_id] = []
            pred_pts_uv_json_save[lanes_id] = []
            for i in range(pred_pts_uv[lanes_id].shape[0]):
                if (pred_pts_uv[lanes_id][i, 0]>1220) or (pred_pts_uv[lanes_id][i, 0]<0) or (pred_pts_uv[lanes_id][i, 1]>370) or (pred_pts_uv[lanes_id][i, 1]<200):
                    continue
                pred_pts_uv_json[lanes_id].append(pred_pts_uv[lanes_id][i])

                u_value, v_value = pred_pts_uv[lanes_id][i]
                u_value, v_value = int(u_value), int(v_value)
                pred_pts_uv_json_save[lanes_id].append({"u": u_value, "v": v_value})
            pred_pts_uv_json[lanes_id] = np.array(pred_pts_uv_json[lanes_id]).reshape((-1, 2))

            biaozhu_pts_uv_json[lanes_id] = []
            biaozhu_pts_uv_json_save[lanes_id] = []
            for i in range(biaozhu_pts_uv[lanes_id].shape[0]):
                if (biaozhu_pts_uv[lanes_id][i, 0] > 1220) or (biaozhu_pts_uv[lanes_id][i, 0] < 0) or (biaozhu_pts_uv[lanes_id][i, 1] > 370) or (biaozhu_pts_uv[lanes_id][i, 1] < 200):
                    continue
                biaozhu_pts_uv_json[lanes_id].append(biaozhu_pts_uv[lanes_id][i])

                u_value, v_value = biaozhu_pts_uv[lanes_id][i]
                u_value, v_value = int(u_value), int(v_value)
                biaozhu_pts_uv_json_save[lanes_id].append({"u": u_value, "v": v_value})
            biaozhu_pts_uv_json[lanes_id] = np.array(biaozhu_pts_uv_json[lanes_id]).reshape((-1, 2))

        print("pred_pts_uv_json: ", pred_pts_uv_json)
        print("pred_pts_uv_json_save: ", pred_pts_uv_json_save)
        print("biaozhu_pts_uv_json: ", biaozhu_pts_uv_json)
        print("biaozhu_pts_uv_json_save: ", biaozhu_pts_uv_json_save)

        save_pred_json = os.path.join(save_dir_frame, sequences_id + "_" + id + "_pred_pts_uv.json")
        save_biaozhu_json = os.path.join(save_dir_frame, sequences_id + "_" + id + "_biaozhu_pts_uv.json")
        json.dump(pred_pts_uv_json_save, open(save_pred_json, "w"))
        json.dump(biaozhu_pts_uv_json_save, open(save_biaozhu_json, "w"))


        pred_pts_color = (255, 0, 0)
        biaozhu_pts_color = (0, 0, 255)
        points_size = 1
        thickness = 1
        for lanes_id in lanes_ids:
            # 依次绘制每一条车道线


            # 重建点云上的标注（自己标注的结果）
            pred_pts_uv_start = pred_pts_uv_json[lanes_id][:-1]
            #print("pred_pts_uv_start: ", pred_pts_uv_start)
            #print("shape of pred_pts_uv_start", pred_pts_uv_start.shape)
            pred_pts_uv_end = pred_pts_uv_json[lanes_id][1:]
            #print("pred_pts_uv_end: ", pred_pts_uv_end)
            #print("shape of pred_pts_uv_end: ", pred_pts_uv_end.shape)
            for i in range(pred_pts_uv_start.shape[0]):
                start = pred_pts_uv_start[i]
                end = pred_pts_uv_end[i]
                cv2.line(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), pred_pts_color, thickness)

            biaozhu_pts_uv_start = biaozhu_pts_uv_json[lanes_id][:-1]
            biaozhu_pts_uv_end = biaozhu_pts_uv_json[lanes_id][1:]
            for i in range(biaozhu_pts_uv_start.shape[0]):
                start = biaozhu_pts_uv_start[i]
                end = biaozhu_pts_uv_end[i]
                cv2.line(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), biaozhu_pts_color, thickness)

        save_jpg_path = os.path.join(save_dir_frame, sequences_id + "_" + id + "_label.jpg")
        cv2.imwrite(save_jpg_path, img)

        #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        #plt.savefig(save_jpg_path, dpi=600)
            #plt.show()














