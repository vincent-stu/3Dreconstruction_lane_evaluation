import numpy as np
import open3d as o3d 

# 标注的点云数据文件路径
point_flag = r"F:\三维重建(毫末公司实习）\data\get_label\00005_0.txt"
# 预测的点云数据文件路径
point_pred = r"F:\三维重建(毫末公司实习）\data\haomo_4dlabel\lidar_label\ds_sampler_00005.txt"
save_txt = r"F:\三维重建(毫末公司实习）\data\haomo_4dlabel\lidar_label\lidar_00005_lane1.txt"

# 处理标注的点云数据（车道线）
data_flag = []
with open(point_flag, 'r') as f:
    for line in f.readlines():
        line = line.strip("\n")
        line = line.split(",")
        line = [float(x) for x in line]
        data_flag.append(line[:3])
points_flag = np.array(data_flag).reshape(-1, 3)
print("shape of points_flag: ", points_flag.shape)
print("points_flag:  ", points_flag)

# 处理预测的点云数据（车道线）
#points_pred = np.loadtxt(point_pred)
#print("type of points_pred:  ", type(points_pred))
#print("shape of points_pred:  ", points_pred.shape)
#print("data_pred: ", points_pred)


pcd_flag = o3d.geometry.PointCloud()
pcd_flag.points = o3d.utility.Vector3dVector(points_flag[:, 0:3])
#pcd_flag.paint_uniform_color([1, 0, 0])

#pcd_pred = o3d.geometry.PointCloud()
#pcd_pred.points = o3d.utility.Vector3dVector(points_pred)
#pcd_pred.paint_uniform_color([0, 0, 1])


vis = o3d.visualization.VisualizerWithVertexSelection()
vis.create_window(window_name='Open3D', visible=True)
#vis.add_geometry(pcd_pred)
vis.add_geometry(pcd_flag)
#vis.add_geometry(pcd_pred)
vis.run()
point = vis.get_picked_points()
vis.destroy_window()

result = []
for i in range(len(point)):
    result.append(point[i].coord)

result = np.array(result).reshape(-1, 3)
print(result)
np.savetxt(save_txt, result)


