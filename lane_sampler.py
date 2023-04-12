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


y_start = np.asarray([[-2.407, 3.991, 50.861], [1.749, 3.929, 48.411]])
y_end = np.asarray([[-2.526, 2.884, 26.940], [1.550, 2.884, 26.963]])
step = 500
y_inter = (y_end-y_start)/step
seq = np.arange(0, step, 1)

seq = seq[:, None, None].repeat(y_start.shape[0], axis=1).repeat(y_start.shape[1], axis=2)
y_points = seq*y_inter[None].repeat(step, axis=0)+y_start[None].repeat(step, axis=0)
#y_points = y_points.reshape(-1, 3)

pred = {}
pred['lane0'] = []
pred['lane1'] = []
#pred['lane2'] = []
for i in range(step):
    x, y, z = y_points[i, 0, :]
    pred['lane0'].append({"x":x, "y":y, "z":z})
    x, y, z = y_points[i, 1, :]
    pred['lane1'].append({"x":x, "y":y, "z":z})
    #x, y, z = y_points[i, 2, :]
    #pred['lane2'].append({"x":x, "y":y, "z":z})

save_dir ="/root/vincent/Scenerf/scenerf_result_json/00046/"    
json.dump(pred, open(save_dir + 'lane_gt_00046.json', 'w'))


#np.savetxt(save_dir + 'gt_sampler_00005.txt', y_points)


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