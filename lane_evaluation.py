import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def point_distance_line(x1, y1, x2, y2, label):
    # x1, y1 曲线2
    # x2, y2 曲线1
    val = dict(zip(x1, y1))
    dis_list = []
    for point in np.array(tuple(zip(x2, y2))):
        xx = point[0]
        zzy = point[1]
        val_list = list(x1)
        val_list.append(xx)  # 将曲线1上的点添加到曲线点中，找到距离该点最近的两个点
        sort_val = sorted(val_list)
        # 如果当前点是最后一个点，取其前两个点
        if sort_val.index(xx) == len(val_list) - 1:
            lind = sort_val.index(xx) - 1
            rind = sort_val.index(xx) - 2
        # 如果当前点是第一个点，取其后两个点
        elif sort_val.index(xx) == 0:
            lind = sort_val.index(xx) + 1
            rind = sort_val.index(xx) + 2
        # 否则，取其前一个点和后一个点
        else:
            lind = sort_val.index(xx) - 1
            rind = sort_val.index(xx) + 1

        plx = sort_val[lind]
        prx = sort_val[rind]
        ply = val[plx]
        pry = val[prx]
        line_point1 = np.array([plx, ply])
        line_point2 = np.array([prx, pry])

        mfy = ply + ((xx - plx) * (pry - ply)) / (prx - plx)   # 曲线1横坐标代入曲线2对应直线后得到的纵坐标

        # 计算向量
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        # 求距离
        distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
        # label等于0 ,曲线2纵坐标小于曲线1纵坐标
        if label == 0:
            if zzy >= mfy:
                dis_list.append(distance)
            else:
                dis_list.append(-distance)
        # label等于1  曲线2纵坐标大于曲线1纵坐标
        elif label == 1:
            if zzy <= mfy:
                dis_list.append(distance)
            else:
                dis_list.append(-distance)
    return dis_list


def eva_line(dt_dir, gt_dir):

    #读取json文件
    with open(dt_dir, 'r') as dt_file:
        dt_json = json.loads(dt_file.read())
    with open(gt_dir, 'r') as gt_file:
        gt_json = json.loads(gt_file.read())

    dt_lines = []
    gt_lines = []
    for dt_key, dt_value in dt_json.items():
        dt_line_tem = []
        for item in dt_value:
            point_tem = (item.get("x"), item.get("z"))
            dt_line_tem.append(point_tem)
        dt_lines.append(dt_line_tem)

    for gt_key, gt_value in gt_json.items():
        gt_line_tem = []
        for gt_item in gt_value:
            point_tem_gt = (gt_item.get("x"), gt_item.get("z"))
            gt_line_tem.append(point_tem_gt)
        gt_lines.append(gt_line_tem)


    #依次处理每一条车道线数据
    for i in range(len(dt_lines)):
        x = list()
        z = list()
        for point in dt_lines[i]:
            temp1, temp2 = point
            x.append(temp1)
            z.append(temp2)
        x = np.array(x)
        z = np.array(z)
        #为方便绘图，拟合出函数表达式
        coef_dt = np.polyfit(x, z, 4)
        func_dt = np.poly1d(coef_dt)

        a = list()
        b = list()
        for point in gt_lines[i]:
            temp1, temp2 = point
            a.append(temp1)
            b.append(temp2)
        a = np.array(a)
        b = np.array(b)
        # 为方便绘图，拟合出函数表达式
        coef_gt = np.polyfit(a, b, 4)
        func_gt = np.poly1d(coef_gt)

        interval_dt = np.arange(min(x), max(x), 0.1)
        interval_gt = np.arange(min(a), max(a), 0.1)

        y_gt = func_gt(interval_gt)
        y_dt = func_dt(interval_dt)

        # 设置中文字体
        mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
        mpl.rcParams['font.size'] = 12  # 字体大小
        mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        plt.plot(interval_gt, y_gt, 'r', label='ground truth')
        plt.plot(interval_dt, y_dt, 'b', label='predicted values')
        plt.xlabel("x坐标值(单位m)")
        plt.ylabel("y坐标值(单位m)")
        plt.title("车道线数据对比图")
        plt.legend()
        plt.savefig("./evaluation_image/lane{}_evaluation.jpg".format(i), dpi = 600)
        plt.show()

        distance = point_distance_line(interval_dt, y_dt, interval_gt, y_gt, label=0)
        distance = [abs(x) for x in distance]
        print("车道线{}的精度(m): ".format(i), sum(distance) / len(distance))



if __name__ == '__main__':
    pred_path = './lane_pred.json'
    gt_path = './lane_gt.json'
    eva_line(pred_path, gt_path)



