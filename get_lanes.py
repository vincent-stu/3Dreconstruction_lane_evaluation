import json
import numpy as np
import math
import cv2
import os
import matplotlib.pyplot as plt
import argparse
import os

def config():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--card_file', type=str, default='/root/fengyuan/63a405bde00bd020df909f7a.txt')
    parser.add_argument('--card_file', type=str, default='/root/vincent/data/get_label/label_paths.txt')
    #parser.add_argument('--pose_file', type=str, default='/root/vincent/data/haomo_4dlabel/poses/')
    parser.add_argument('--result_dir', type=str, default='/root/vincent/data/haomo_4dlabel/lidar_label')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1000000)
    return parser.parse_args()

def rpy2rotation(theta, format='degree'):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

sequences_id = ["00005", "00014", "00015", "00017", "00021", "00031", "00042"]
poses_file = "/root/vincent/data/haomo_4dlabel/poses/"

if __name__ == '__main__':
    args = config()
    
    for id, json_file in enumerate(open(args.card_file)):

        if id % 100 == 0:
            save_dir = os.path.join(args.result_dir, sequences_id[id // 100])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)


        json_file = json_file.strip("\n")
        arr = json.load(open(json_file, 'r'))
        hardware_arr = json.load(open('/' + arr['hardware_config_path'], 'r'))

        for cam in hardware_arr['sensor_config']['cam_param']:
            if 'name' in cam and cam['name'] == 'front_middle_camera':
                cam_pose = cam['pose']
                cam_cx = cam['cx']
                cam_cy = cam['cy']
                cam_fx = cam['fx']
                cam_fy = cam['fy']
                cam_distortion = cam['distortion']
                break
        for lidar in hardware_arr['sensor_config']['lidar_param']:
            if lidar['name'] == 'MIDDLE_LIDAR':
                lidar_pose = lidar['pose']
                break
            
        cam_intrinsic = np.eye(3, 4)
        cam_intrinsic[0][0] = cam_fx
        cam_intrinsic[1][1] = cam_fy
        cam_intrinsic[0][2] = cam_cx
        cam_intrinsic[1][2] = cam_cy

        #cam 2 car
        cam_rpy = [cam_pose['attitude_ypr']['roll'],cam_pose['attitude_ypr']['pitch'],\
                cam_pose['attitude_ypr']['yaw']]
        cam2car_tx,cam2car_ty,cam2car_tz = cam_pose['translation']['x'], cam_pose['translation']['y'],\
            cam_pose['translation']['z']
        cam2car = np.eye(4,4)
        cam2car_rotation = rpy2rotation(cam_rpy)
        cam2car[:3,:3] = cam2car_rotation
        cam2car[:3,3] = np.array([cam2car_tx,cam2car_ty,cam2car_tz]).T

        #lidar 2 cam
        lidar_rpy = [lidar_pose['attitude_ypr']['roll'], lidar_pose['attitude_ypr']['pitch'],\
            lidar_pose['attitude_ypr']['yaw']]
        lidar2car_tx,lidar2car_ty,lidar2car_tz = lidar_pose['translation']['x'],0.,\
            lidar_pose['translation']['z']
        lidar2car = np.eye(4,4)
        lidar2car_rotation = rpy2rotation(lidar_rpy)
        lidar2car[:3,:3] = lidar2car_rotation
        lidar2car[:3,3] = np.array([lidar2car_tx,lidar2car_ty,lidar2car_tz]).T

        lidar2cam = np.linalg.inv(cam2car) @ lidar2car
        
        points = []

        points_start = []
        points_end = []
        for obj in arr['labeled_data']['objects']:
            if obj['class_name'] == 'lane':
                for child in obj['children']:
                    start = child['geometry']['points'][0]
                    for point in child['geometry']['points'][1:]:
                        x, y, z = start['x'], start['y'], start['z']
                        points_start.append([x, y, z])
                        x, y, z = point['x'], point['y'], point['z']
                        points_end.append([x, y, z])
                        start = point
        points_start = np.vstack(points_start)
        points_end = np.vstack(points_end)
        points = np.concatenate([points_start, points_end], axis=0)
        print("shape of points:  ", points.shape)
        print("points: ", points)
        
        
        bundle_path = '/' + arr['bundle_oss_path']
        bundle = json.load(open(bundle_path, 'r'))

        img_path = '/' + bundle['camera'][1]['oss_path']
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        distortion = np.array([cam_distortion[0], cam_distortion[1], cam_distortion[3], cam_distortion[4], cam_distortion[2]])
        img = cv2.undistort(img, cam_intrinsic[:3, :3], distortion)

        I = cam_intrinsic[:3, :3]
        E = lidar2cam

        points_ = np.concatenate([points, np.ones_like(points[:, :1])], axis=1)

        pose_path = os.path.join(poses_file, sequences_id[id // 100] + ".txt")
        with open(pose_path, 'r') as f:
            pose_data = f.readlines()
        
        pose = [float(x) for x in pose_data[id % 100].strip("\n").split(" ")]
    
        pose = np.array(pose)
        pose = pose.reshape(3, 4)
        pose = np.vstack((pose, [0.0, 0.0, 0.0, 1.0]))
        
        points_cam = E @ points_.T
        points_world = pose @ points_cam
        points_world = points_world.T
        

        points_pix = I @ points_cam[:3]
        depth = points_pix.T[:, 2:]
        uv = points_pix.T[:, :2] / depth
        l = uv.shape[0]
        uv_start = uv[:l//2]
        uv_end = uv[l//2:]
        depth_start = depth[:l//2]
        depth_end = depth[l//2:]
        h,w,_= img.shape

        vis_index = np.where((uv_start[:,0]>0) & (uv_start[:,0]<w) & (uv_start[:,1]>0)& (uv_start[:,1]<h)&(depth_start[:,0]>0)&(depth_start[:,0]<100)&(uv_end[:,0]>0) & (uv_end[:,0]<w) & (uv_end[:,1]>0)& (uv_end[:,1]<h)&(depth_end[:,0]>0)&(depth_end[:,0]<100))
        uv_vis = uv[vis_index]

        point_color = (255, 0, 0)
        point_size = 5
        thickness = 5

        
        save_name_txt = sequences_id[id // 100] + "_" + str(id % 100) + ".txt"
        print("save_path:  ", os.path.join(save_dir, save_name_txt))
        np.savetxt(os.path.join(save_dir, save_name_txt), points_world, delimiter=',')


        for index in vis_index[0]:
            start = uv_start[index]
            end = uv_end[index]
            cv2.line(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), point_color, thickness)
        save_name_jpg = sequences_id[id//100] + "_" + str(id % 100)  + ".jpg"
        save_path = os.path.join(save_dir, save_name_jpg)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img)     