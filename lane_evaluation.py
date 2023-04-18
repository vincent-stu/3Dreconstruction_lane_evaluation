import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def calc_point_distance(pt1, pt2):
    # 计算两点之间的距离
    dist = math.sqrt(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2))
    return dist


def fit_curve(lanes, m):
    # 拟合车道线数据
    # m: 表示多项式的阶数

    lanes_func = []  # 存放每一条车道线的拟合函数
    for i in range(len(lanes)):
        # 依次处理每一条车道线数据
        x = list()
        z = list()
        for point in lanes[i]:
            temp1, temp2 = point
            x.append(temp1)
            z.append(temp2)
        x = np.array(x)
        z = np.array(z)
        #为方便绘图，拟合出函数表达式
        coef = np.polyfit(z, x, m)
        func = np.poly1d(coef)
        lanes_func.append(func)

    return lanes_func

def eval_lanes(ds_dir, gt_dir, save_dir, m):

    #读取json文件
    with open(ds_dir, 'r') as ds_file:
        ds_json = json.loads(ds_file.read())
    with open(gt_dir, 'r') as gt_file:
        gt_json = json.loads(gt_file.read())

    ds_lines = []
    gt_lines = []
    for ds_key, ds_value in ds_json.items():
        ds_line_tem = []
        for item in ds_value:
            point_tem = (item.get("x"), item.get("z"))
            ds_line_tem.append(point_tem)
        ds_lines.append(ds_line_tem)
        #对每一条车道线按照z坐标值进行降序排序
    for line_a in ds_lines:
        line_a.sort(reverse=True, key = lambda x: x[1])

    for gt_key, gt_value in gt_json.items():
        gt_line_tem = []
        for gt_item in gt_value:
            point_tem_gt = (gt_item.get("x"), gt_item.get("z"))
            gt_line_tem.append(point_tem_gt)
        gt_lines.append(gt_line_tem)
        #对每一条车道线按照z坐标值进行降序排序
    for line_b in gt_lines:
        line_b.sort(reverse=True, key = lambda x: x[1])
    #print("ds_lines:  ", ds_lines)
    #print("gt_lines:  ", gt_lines)

    ds_lines_func = fit_curve(ds_lines, m)
    gt_lines_func = fit_curve(gt_lines, m)
    #print("ds_lines_func: ", ds_lines_func)
    #print("gt_lines_func ", gt_lines_func)
    for i in range(len(ds_lines_func)):

        #依次处理每一条车道线
        ds_z_min = ds_lines[i][-1][1]
        #print('ds_z_min: ', ds_z_min)
        ds_z_max = ds_lines[i][0][1]
        #print("ds_z_max: ", ds_z_max)
        gt_z_min = gt_lines[i][-1][1]
        #print("gt_z_min: ", gt_z_min)
        gt_z_max = gt_lines[i][0][1]
        #print("gt_z_max: ", gt_z_max)

        if (gt_z_max <= ds_z_min) or (gt_z_min >= ds_z_max):
            print("Warning: 采样不规范！！")
            
            dist = []
            
            #print("ds_z_min: ", ds_z_min)
            #print("ds_z_max: ", ds_z_max)
            z_interval = np.arange(ds_z_min, ds_z_max, 0.1)
            #print("z_interval:  ", z_interval)
            x_ds = ds_lines_func[i](z_interval)
            x_gt = gt_lines_func[i](z_interval)
            #print("x_ds: ", x_ds)
            #print("x_gt: ", x_gt)
            for j in range(len(z_interval)):
                dist_temp = float('inf')
                for k in range(len(z_interval)):
                    ds_point = [x_ds[j], z_interval[j]]
                    gt_point = [x_gt[k], z_interval[k]]
                    point_dist = calc_point_distance(ds_point, gt_point)
                    if point_dist < dist_temp:
                        dist_temp = point_dist

                dist.append(dist_temp)
            #print("dist: ", dist)
        else:
            dist = []
            z_min = max(ds_z_min, gt_z_min)
            z_max = min(ds_z_max, gt_z_max)
            z_interval = np.arange(z_min, z_max, 0.1)
            #print("z_interval: ", z_interval)
            #print("len of z_interval: ", len(z_interval))
            x_ds = ds_lines_func[i](z_interval)
            x_gt = gt_lines_func[i](z_interval)
            #print("len of x_ds: ", len(x_ds))
            #print("len of x_gt: ", len(x_gt))
            #print("x_ds: ", x_ds)
            #print("x_gt: ", x_gt)
            for j in range(len(z_interval)):
                dist_temp = float('inf')
                #print(j)
                for k in range(len(z_interval)):
                    ds_point = [x_ds[j],  z_interval[j]]
                    gt_point = [x_gt[k],  z_interval[k]]
                    point_dist = calc_point_distance(ds_point, gt_point)
                    #print("point_dist: ", point_dist)
                    if point_dist < dist_temp:
                        dist_temp = point_dist
                    #print("dist_temp: ", dist_temp)

                dist.append(dist_temp)
                #print("dist:  ", dist)
            #print("len of dist: ", len(dist))
            #print("dist: ", dist)

        # 绘制图像
        plt.figure(figsize=(16, 9))
        plt.plot(z_interval, x_gt, 'r', label='ground truth')
        plt.plot(z_interval, x_ds, 'b', label='predicted values')
        plt.xlabel("Z coordinate value(m)")
        plt.ylabel("X coordinate value(m)")
        plt.title("Lane {} Comparison".format(i))
        plt.legend()

        #plt.savefig(save_dir + "baiozhu_lane{}_evaluation.jpg".format(i), dpi=600)
        plt.show()
        print("车道线{}的精度(m): ".format(i), sum(dist) / len(dist))

        
if __name__ == '__main__':
    save_dir ='/root/vincent/Scenerf/scenerf_result_json/00046/'
    pred_path = '/root/vincent/Scenerf/scenerf_result_json/00046/lane_ds_00046.json'
    #gt_path = '/root/vincent/Scenerf/scenerf_result_json/00046/lane_gt_00046.json'
    gt_path = '/root/vincent/Scenerf/scenerf_result_json/00046/lane_00046_biaozhu.json'

    #雷达点云做参考进行评估时，由于数据量比较多，可以选择高阶的多项式进行拟合(例如：4)。
    #而当采用人工标注作为参考进行评估时，由于数据量比较少，并且由于车道线大部分情况下可近似为直线，因此应该选择低阶的多项式进行拟合(例如：1)。
    eval_lanes(pred_path, gt_path, save_dir, 1)     

