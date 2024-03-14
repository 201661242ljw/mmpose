import copy
import shutil

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json

from mmpose.utils import ljw_tower_pose_pack_accuracy, ljw_tower_skeleton_accuracy


def see__or_oks():
    # 计算Z值
    S = 256
    sigma = 15
    w1 = w2 = h1 = h2 = sigma * 3
    x1 = 1000
    y1 = 1000

    thresh = 0.5
    # 定义横纵坐标范围和步长
    x = np.arange(2 * x1)
    y = np.arange(2 * y1)

    # 生成网格
    X, Y = np.meshgrid(x, y)

    scale = 100

    X = X[x1 - scale:x1 + scale, y1 - scale:y1 + scale]
    Y = Y[x1 - scale:x1 + scale, y1 - scale:y1 + scale]

    left = np.maximum(x1, X)
    top = np.maximum(y1, Y)
    right = np.minimum(x1 + w1, X + w2)
    bottom = np.minimum(y1 + h1, Y + h2)

    c1 = right > left
    c2 = bottom > top

    c = c1.astype(float) * c2.astype(float)
    intersection = (right - left) * (bottom - top)
    union = w1 * h1 + w2 * h2 - intersection

    c = c * intersection / union

    dd = (x1 - X) * (x1 - X) + (y1 - Y) * (y1 - Y)
    oks = np.exp(-(dd / 1 / sigma / sigma))

    c = ((c > thresh).astype(int) * 255).astype(np.uint8)

    oks = ((oks > thresh).astype(int) * 255).astype(np.uint8)

    print(np.sum(c))
    print(np.sum(oks))

    img = cv2.merge([c * 0, c, oks])

    cv2.imwrite(r"oks_iou.jpg", img)

    # 创建图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scale = 10
    # 绘制3D图
    # ax.plot_surface(X[x1 - scale:x1 + scale, y1 - scale:y1 + scale], Y[x1 - scale:x1 + scale, y1 - scale:y1 + scale],
    #                 c[x1 - scale:x1 + scale, y1 - scale:y1 + scale], cmap='viridis')  # 绘制c的三维图
    # ax.plot_surface(X[x1 - scale:x1 + scale, y1 - scale:y1 + scale], Y[x1 - scale:x1 + scale, y1 - scale:y1 + scale],
    #                 oks[x1 - scale:x1 + scale, y1 - scale:y1 + scale], cmap='plasma')  # 绘制oks的三维图
    #
    # # 设置坐标轴标签
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    #
    # # 显示图形
    # plt.show()


def rename_points_ue():
    src_img_dir_all = r"../data/UE4/backup/IMG_"
    dst_img_dir_all = r"../data/UE4/IMG_"

    src_json_dir_all = r"../data/UE4/backup/json"

    for type_ in ["3", "7"]:
        src_img_dir = r"../data/UE4/{}_img".format(type_)
        dst_json_dir = r"../data/UE4/{}_json".format(type_)
        for img_name in os.listdir(src_img_dir):
            src = os.path.join(src_img_dir_all, img_name)
            dst = os.path.join(dst_img_dir_all, img_name)
            if not os.path.exists(dst):
                shutil.copy(src, dst)

                json_path = os.path.join(src_json_dir_all, "{}.json".format(img_name.split(".")[0]))
                f = open(json_path, "r", encoding="utf-8").read()
                f = f.replace("edge", f"{type_}_edge").replace("sk", f"{type_}_sk").replace("bi", f"{type_}_bi")
                dst_json_path = os.path.join(dst_json_dir, "{}.json".format(img_name.split(".")[0]))
                f2 = open(dst_json_path, "w", encoding="utf-8")
                f2.write(f)
                f2.close()


def copy_json():
    src_dir = r"E:\LJW\Git\mmpose\tools\0_LJW_tools\predict_show\00_json_out"

    json_path = r"img_name_2_type.json"

    data = json.load(open(json_path, "r", encoding="utf-8"), strict=False)
    id_2_name = data['id_2_name']
    id_2_type = data['id_2_type']

    dst_dir = r"E:\LJW\Git\mmpose\tools\0_LJW_tools\predict_show\00_json_out_2"
    for file_name in os.listdir(src_dir):
        src = os.path.join(src_dir, file_name)
        dst_name = id_2_name[file_name.split(".")[0]]
        a = 1
        dst = os.path.join(dst_dir, dst_name.split(".")[0] + '.json')
        if not os.path.exists(dst):
            shutil.copy(src, dst)


def test_results(file_path):
    import copy
    sigma = 1.5 * 4 * 2
    iou_threshhold = 0.5

    data = json.load(open(file_path, "r", encoding="utf-8"), strict=False)
    kt3 = data['kt3']
    pred_kt_3 = kt3['pred']
    gt_kt_3 = kt3['gt']
    tp, fp, fn = ljw_tower_pose_pack_accuracy(pred_kt_3,
                                              copy.deepcopy(gt_kt_3),
                                              sigma,
                                              iou_threshold=iou_threshhold)
    print(tp, fp, fn)


def fix_ue_dataset():
    shift_list = [
        3,
        4,
        1,
        2,
        7,
        8,
        5,
        6
    ]
    for tower_type in ['3', '7']:
        src_json_dir = r"../data/UE4/{}_json_".format(tower_type)
        dst_json_dir = r"../data/UE4/{}_json".format(tower_type)
        for file_name in os.listdir(src_json_dir):
            json_path = os.path.join(src_json_dir, file_name)
            data = json.load(open(json_path, "r", encoding="utf-8"), strict=False)
            pts = data['points']
            new_data = copy.deepcopy(data)
            if pts[f'{tower_type}_edgepoint_1_1_']['x'] > pts[f'{tower_type}_edgepoint_1_2_']['x'] :
                for (p_name, temp_dict) in pts.items():
                    [t_type,p_type,floor,position,_] = p_name.split("_")
                    if p_type != "skeletonpoint":
                        new_position = ['2','1'][['1','2'].index(position)]
                    elif t_type == "7" and p_type=="bileipoint":
                        new_position = '1'
                    else:
                        new_position = shift_list[int(position) - 1]
                    new_name = f"{t_type}_{p_type}_{floor}_{new_position}_"
                    new_data['points'][new_name] = temp_dict
            save_path = os.path.join(dst_json_dir, file_name)
            b = json.dumps(new_data, ensure_ascii=False, indent=4)
            f2 = open(save_path, 'w', encoding='utf-8')
            f2.write(b)
            f2.close()

        for file_name in os.listdir(dst_json_dir):
            json_path = os.path.join(dst_json_dir, file_name)
            data = json.load(open(json_path, "r", encoding="utf-8"), strict=False)
            pts = data['points']
            if pts[f'{tower_type}_edgepoint_1_1_']['x'] > pts[f'{tower_type}_edgepoint_1_2_']['x'] :
                print(file_name)

            a = 1

if __name__ == '__main__':
    # path = os.getcwd()
    # print(path)
    # rename_points_ue()
    # copy_json()

    # file_name = r"scene_2023-11-21_10-34-49_995.json"
    # file_path = r"E:\LJW\Git\mmpose\tools\0_LJW_tools\predict_show\00_json_out_2\{}".format(file_name)
    # test_results(file_path)
    fix_ue_dataset()
