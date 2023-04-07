import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def read_poses(path):
    # Read and parse the poses
    poses = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
            T_w_cam0 = T_w_cam0.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            poses.append(T_w_cam0)
    return poses


def get_depth_from_lidar(lidar_path, P, T_velo_2_cam, image_size):
    scan = np.fromfile(lidar_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, :3]
        
    keep_idx = points[:, 0] > 0  # only keep point in front of the vehicle
    points_hcoords = np.concatenate([points[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)
       
    pts_cam = (T_velo_2_cam @ points_hcoords.T).T
    # print(pts_cam[:, 2].min(), pts_cam[:, 2].max())
    mask = (pts_cam[:, 2] <= 80) & (pts_cam[:, 2] > 0)  # get points with depth < max_sample_depth
    pts_cam = pts_cam[mask, :3]

    img_points = (P[0:3, 0:3] @ pts_cam.T).T

    img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
    img_points = np.round(img_points).astype(int)
        
    keep_idx_img_pts = (img_points[:, 0] > 0) & \
                        (img_points[:, 1] > 0) & \
                        (img_points[:, 0] < image_size[0]) & \
                        (img_points[:, 1] < image_size[1])
                           
    img_points = img_points[keep_idx_img_pts, :]
        
    pts_cam = pts_cam[keep_idx_img_pts, :]

    depths = pts_cam[:, 2]

    return img_points, depths, pts_cam


calib_path = '/mnt/ve_share/xwy/data/haomo2kitti_camera_center_undistort/sequences/00/calib.txt' 
calib_all = {}
with open(calib_path, "r") as f:
    for line in f.readlines():
        if line == "\n":
            break
        key, value = line.split(":", 1)
        calib_all[key] = np.array([float(x) for x in value.split()])
calib_out = {}
# 3x4 projection matrix for left camera
calib_out["P0"] = calib_all["P0"].reshape(3, 4)
calib_out["Tr"] = np.identity(4)  # 4x4 matrix
calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)

# depth_dir3 = "/root/vincent/Scenerf/scenerf_result/recon_haomo_kittipretrain_1220x370_try/depth/00"
depth_dir2 = "/root/vincent/scenerf_result/recon_haomo_kittipretrain_1220x370_2/depth/00"
# depth_dir1 = "/root/vincent/scenerf_result/recon_haomo_kittipretrain_1220x370/depth/00"
error_sum1 = []   # 全图平均绝对误差
error_sum2 = []   # 全图平均相对误差
error_sum3 = []   # depth 15米以内的平均绝对误差
error_sum4 = []   # depth 15米以内的平均相对误差
error_sum5 = []   # 人为指定区域内的平均绝对误差
error_sum6 = []   # 人为指定区域内的平均相对误差
coef1 = []   # 全图范围内的系数
coef2 = []   # depth 15米以内的系数
coef3 = []   # 人为指定区域内的系数

for x in os.listdir(depth_dir2):
    id = x.split(".")[0] 
    error_temp1 = []
    error_temp2 = []
    error_temp3 = []
    error_temp4 = []
    error_temp5 = []
    error_temp6 = []
    img_path = '/root/vincent/scenerf_result/recon_haomo_kittipretrain_1220x370_2/rgb/00/{}.png'.format(id)
    # if not os.path.exists(img_path):
    #    continue 

    img = cv2.imread(img_path)
    depth = np.load('/root/vincent/scenerf_result/recon_haomo_kittipretrain_1220x370_2/depth/00/{}.npy'.format(id))
    I = np.load('/root/vincent/scenerf_result/recon_haomo_kittipretrain_1220x370_2/I/00/{}.npy'.format(id))
    # E = np.load('/root/vincent/Scenerf/scenerf_result/recon_haomo_kittipretrain_1220x370_try/E/00/{}.npy'.format(id))
    lidar_path = '/mnt/ve_share/xwy/data/haomo2kitti_camera_center_undistort/sequences/00/velodyne/{}.bin'.format(id)

    h, w, c = img.shape

    loc2d_with_depth, lidar_depth, points_cam = get_depth_from_lidar(lidar_path, I, calib_out['Tr'], (w, h))
    # lidar_depth[:,np.newaxis]
    for i in range(loc2d_with_depth.shape[0]):
        depth_temp = depth[loc2d_with_depth[i, 1], loc2d_with_depth[i, 0]]

        error_temp1.append(abs(lidar_depth[i] - depth_temp))  
        error_temp2.append(abs(lidar_depth[i] - depth_temp)/abs(lidar_depth[i])) 
        coef1.append(abs(lidar_depth[i]/depth_temp))

        if depth_temp <= 15:
            error_temp3.append(abs(lidar_depth[i] - depth_temp))
            error_temp4.append(abs(lidar_depth[i] - depth_temp)/abs(lidar_depth[i]))
            coef2.append(abs(lidar_depth[i]/depth_temp))
        if loc2d_with_depth[i, 1]>=280 and (loc2d_with_depth[i, 0] <= 900 and loc2d_with_depth[i, 0] >= 400):
            error_temp5.append(abs(lidar_depth[i] - depth_temp))
            error_temp6.append(abs(lidar_depth[i] - depth_temp)/abs(lidar_depth[i]))
            coef3.append(abs(lidar_depth[i]/depth_temp))
        

    error_sum1.append(sum(error_temp1)/len(error_temp1))
    error_sum2.append(sum(error_temp2)/len(error_temp2))
    error_sum3.append(sum(error_temp3)/len(error_temp3))
    error_sum4.append(sum(error_temp4)/len(error_temp4))
    error_sum5.append(sum(error_temp5)/len(error_temp5))
    error_sum6.append(sum(error_temp6)/len(error_temp6))


coef1 = np.array(coef1)
coef2 = np.array(coef2)
coef3 = np.array(coef3)
# 去除异常值
coef1 = np.clip(coef1, 0, np.percentile(coef1, 99.9))
coef2 = np.clip(coef2, 0, np.percentile(coef2, 99.9))
coef3 = np.clip(coef3, 0, np.percentile(coef3, 99.9))

# 作图
plt.figure(figsize=(15, 10))
plt.style.use("seaborn-white")
plt.hist(coef1, bins=100, color='r', alpha=0.3, density=True, label="full_image")
plt.hist(coef2, bins=100, color='g', alpha=0.3, density=True, label='depth<=15m')
plt.hist(coef3, bins=100, color='b', alpha=0.3, density=True, label='road_zone')
plt.legend(loc='upper right', frameon=True)
plt.title("scale distribution")
plt.xlabel("scale")
plt.ylabel("frequency / interval")
# plt.axvline(1, c='black', ls='-.')
plt.axvline(1, c='black', ls='-.')
# plt.axvline(1.5, c='black', ls='-.')
plt.savefig("/root/vincent/scenerf_result/recon_haomo_kittipretrain_1220x370_2/不同范围内的scale值分布图.jpg", dpi=1000)


# plt.hist(coef2, bins=100)
# plt.title("scale distribution")
# plt.xlabel("scale")
# plt.ylabel("frequency")
# plt.savefig("depth 15m以内范围的scale值分布图.jpg", dpi=600)

# plt.hist(coef3, bins=100)
# plt.title("scale distribution")
# plt.xlabel("scale")
# plt.ylabel("frequency")
# plt.savefig("部分道路区域范围的scale值分布图.jpg", dpi=600)
    
print("所有图片的全图范围平均绝对误差： {:.3f}m".format(sum(error_sum1)/len(error_sum1)))
print("所有图片的全图范围平均相对误差： {:.3%}".format(sum(error_sum2)/len(error_sum2)))
print("所有图片的depth 15m以内范围的平均绝对误差:  {:.3f}m".format(sum(error_sum3)/len(error_sum3)))
print("所有图片的depth 15m以内范围的平均相对误差:  {:.3%}".format(sum(error_sum4)/len(error_sum3)))
print("所有图片的部分道路区域范围的平均绝对误差： {:.3f}m".format(sum(error_sum5)/len(error_sum5)))
print("所有图片的部分道路区域范围的平均相对误差： {:.3%}".format(sum(error_sum6)/len(error_sum6)))


# point_color = (0, 0, 255)
# point_size = 1
# thickness = 1
# for p in loc2d_with_depth:
#    cv2.circle(img, (int(p[0]), int(p[1])), point_size, point_color, thickness) 

# save_path = '/root/vincent/Scenerf/scenerf_result/recon_haomo_kittipretrain_1220x370_try/rgb_with_lidar/{}.jpg'.format(id)
# cv2.imwrite(save_path, img) 
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()