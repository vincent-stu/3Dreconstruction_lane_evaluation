import matplotlib.pyplot as plt
import json
import numpy as np

# dir = 'rec_result/'
# dir = "/root/vincent/Scenerf/scenerf_result_pointcloud/result_point_cloud/00014_result/"
# im = plt.imread(dir+'img.png')
# E = np.load(dir+'E.npy')
# E = np.linalg.inv(E)
# I = np.load(dir+'I.npy')
# pc = np.loadtxt(dir+'vpoints_gt_00005.txt')

# 总的采样点数
step = 500
# 第一段。如果该场景为直线道路，则仅有这一段。剩余的变量均应为None
y1 = np.asarray([[2.741, 4.312, 61.299], [6.555, 4.353, 60.722]])
y2 = np.asarray([[1.143, 3.835, 51.607], [5.076, 3.891, 51.132]])
# 多段，该场景为曲线道路时使用
y3 = np.asarray([[0.563, 3.555, 46.639], [4.574, 3.611, 46.509]])
y4 = np.asarray([[-0.460, 3.066, 36.297], [3.416, 3.161, 36.131]])
y5 = np.asarray([[-0.936, 2.917, 32.386], [3.036, 2.952, 31.833]])
y6 = np.asarray([[-1.585, 2.428, 21.884], [2.360, 2.416, 21.366]])
# y3 = None
# y4 = None
# y5 = None
# y6 = None

if (y3 is None) and (y4 is None) and (y5 is None) and (y6 is None):
    # 处理直线道路

    y_inter = (y2-y1)/step
    seq = np.arange(0, step, 1)

    seq = seq[:, None, None].repeat(y1.shape[0], axis=1).repeat(y1.shape[1], axis=2)
    y_points = seq*y_inter[None].repeat(step, axis=0)+y1[None].repeat(step, axis=0)
    # print("y_points:  ", y_points)
    # print("shape of y_points:  ", y_points.shape)
    # print("type of y_points: ", type(y_points))


else:

    # 处理弯曲道路

    step_mini = step // 5
    # print(step_mini)

    seq = np.arange(0, step_mini, 1)
    y_inter1 = (y2-y1)/step_mini
    seq = seq[:, None, None].repeat(y1.shape[0], axis=1).repeat(y1.shape[1], axis=2)
    y_point1 = seq*y_inter1[None].repeat(step_mini, axis=0)+y1[None].repeat(step_mini, axis=0)

    seq = np.arange(0, step_mini, 1)
    y_inter2 = (y3-y2)/step_mini
    seq = seq[:, None, None].repeat(y2.shape[0], axis=1).repeat(y2.shape[1], axis=2)
    y_point2 = seq*y_inter2[None].repeat(step_mini, axis=0)+y2[None].repeat(step_mini, axis=0)

    seq = np.arange(0, step_mini, 1)
    y_inter3 = (y4-y3)/step_mini
    seq = seq[:, None, None].repeat(y3.shape[0], axis=1).repeat(y3.shape[1], axis=2)
    y_point3 = seq*y_inter3[None].repeat(step_mini, axis=0)+y3[None].repeat(step_mini, axis=0)

    seq = np.arange(0, step_mini, 1)
    y_inter4 = (y5-y4)/step_mini
    seq = seq[:, None, None].repeat(y4.shape[0], axis=1).repeat(y4.shape[1], axis=2)
    y_point4 = seq*y_inter4[None].repeat(step_mini, axis=0)+y4[None].repeat(step_mini, axis=0)

    seq = np.arange(0, step_mini, 1)
    y_inter5 = (y6-y5)/step_mini
    seq = seq[:, None, None].repeat(y5.shape[0], axis=1).repeat(y5.shape[1], axis=2)
    y_point5 = seq*y_inter5[None].repeat(step_mini, axis=0)+y5[None].repeat(step_mini, axis=0)

    y_points = np.concatenate((y_point1, y_point2, y_point3, y_point4, y_point5), axis=0)
    # print("y_points: ", y_points)
    # print("shape of y_points: ", y_points.shape)
    # print("type of y_points: ", type(y_points))


pred = {}
pred['lane0'] = []
pred['lane1'] = []
# pred['lane2'] = []

for i in range(step):
    x, y, z = y_points[i, 0, :]
    pred['lane0'].append({"x":x, "y":y, "z":z})
    x, y, z = y_points[i, 1, :]
    pred['lane1'].append({"x":x, "y":y, "z":z})
    # x, y, z = y_points[i, 2, :]
    # pred['lane2'].append({"x":x, "y":y, "z":z})

save_dir = "/root/vincent/Scenerf/scenerf_result_json/00049/"    
json.dump(pred, open(save_dir + 'lane_ds_00049.json', 'w'))

y_points = y_points.reshape(-1, 3)
np.savetxt(save_dir + 'ds_sampler_00049.txt', y_points)


# y_points = np.concatenate([y_points, np.ones_like(y_points[:, :1])],axis=1)
# y_cam = E@y_points.T
# y_cam[0]+=cam02cam2
# y_pix = I@y_cam[:3]
# depth = y_pix.T[:,2:]
# uv = y_pix.T[:,:2]/depth
# h,w,_=im.shape
# vis_index =np.where((uv[:,0]>0) & (uv[:,0]<w) & (uv[:,1]>0)& (uv[:,1]<h)&(depth[:,0]>0))
# uv_vis = uv[vis_index]
# plt.imshow(im)
# plt.scatter(uv_vis[:,0],uv_vis[:,1],s=5)
# plt.show()
# a=1