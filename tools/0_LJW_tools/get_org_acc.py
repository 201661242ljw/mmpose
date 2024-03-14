import json

import numpy
import math

import numpy as np


def get_oks(x1, y1, x2, y2, sigma):
    dd = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

    oks = math.exp(-(dd / (2 * sigma * sigma)))
    return oks

oks_threshold= 0.5

gt_json = r"E:\LJW\Git\mmpose\tools\0_LJW_tools\Org_acc\0_keypoints_test.json"
gt_data = json.load(open(gt_json, "r", encoding="utf-8"), strict=False)

pr_json = r"E:\LJW\Git\mmpose\tools\0_LJW_tools\Org_acc\result.json"
pr_data = json.load(open(pr_json, "r", encoding="utf-8"), strict=False)


sigma = 12
new_size = 1024

for oks_threshold in [0.5, 0.75]:
    total_right = 0
    total_wrong = 0

    for i in range(len(pr_data)):
        gt_dict = gt_data['annotations'][i]
        gt_points = np.array(gt_dict['keypoints']).reshape([-1, 3])

        old_xs = gt_points[:, 1].astype(int)
        old_ys = gt_points[:, 2].astype(int)

        x1 = np.min(old_xs)
        x2 = np.max(old_xs)
        width = x2 - x1
        y1 = np.min(old_ys)
        y2 = np.max(old_ys)
        height = y2 - y1

        img_width = gt_data['images'][i]['width']
        img_height = gt_data['images'][i]['height']
        x1 = max(0, int(x1 - width * 0.05))
        x2 = min(img_width - 1, int(x2 + width * 0.05))
        y1 = max(0, int(y1 - height * 0.05))
        y2 = min(img_height - 1, int(y2 + height * 0.05))

        width = x2 - x1
        height = y2 - y1
        s = int(max(width, height) * 1.05)

        gt_pts = (gt_points / s * new_size).astype(int)
        pr_pts = (np.array(pr_data[i]['preds'])[0] / s * new_size).astype(int)

        x1 = gt_pts[:, 0]
        y1 = gt_pts[:, 1]
        x2 = pr_pts[:, 0]
        y2 = pr_pts[:, 1]
        masks = gt_points[:, 2] != 0

        dd = (x1 - x2) ** 2 + (y1 - y2) ** 2

        okss = np.exp(-(dd / (2 * sigma * sigma)))

        right = okss >= oks_threshold
        wrong = okss < oks_threshold

        right_num = np.sum(right.astype(int) * masks)
        wrong_num = np.sum(wrong.astype(int) * masks)

        total_right += right_num
        total_wrong += wrong_num
        # print(right_num)
        # print(wrong_num)
        # print()

    print(total_right)
    print(total_wrong)
    print(total_right / ( total_wrong + total_right))
    print()

