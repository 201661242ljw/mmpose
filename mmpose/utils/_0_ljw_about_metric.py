# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Tuple
import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import math
import colorsys
import cv2
import os


def get_all_peaks(heatmap, sigma, idx_channel=2, upscale=None):
    all_peaks = []
    all_scores = []
    peak_counter = 0
    thre1 = 0.1

    for part in range(heatmap.shape[idx_channel]):
        if idx_channel == 2:
            map_ori = heatmap[:, :, part]
        elif idx_channel == 0:
            map_ori = heatmap[part, :, :]
        one_heatmap = gaussian_filter(map_ori, sigma=sigma)

        map_left = np.zeros(one_heatmap.shape)
        map_left[1:, :] = one_heatmap[:-1, :]
        map_right = np.zeros(one_heatmap.shape)
        map_right[:-1, :] = one_heatmap[1:, :]
        map_up = np.zeros(one_heatmap.shape)
        map_up[:, 1:] = one_heatmap[:, :-1]
        map_down = np.zeros(one_heatmap.shape)
        map_down[:, :-1] = one_heatmap[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up,
             one_heatmap >= map_down, one_heatmap > thre1))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        peak_id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

        if len(peaks_with_score_and_id) != 0 and upscale != None:
            peaks_with_score_and_id = np.array(peaks_with_score_and_id)
            peaks_with_score_and_id[:, :2] *= upscale
            peaks_with_score_and_id = peaks_with_score_and_id.astype(np.int32).tolist()
        all_peaks.append(peaks_with_score_and_id)
        all_scores.append([peaks_with_score[i][-1] for i in range(len(peak_id))])
        peak_counter += len(peaks)
    return all_peaks, all_scores


def get_all_skeletons(paf_ht, all_peaks, img_shape, idx_channel=0, out_form=3, upscale=None):
    thre2 = 0.05

    if idx_channel == 0:
        paf_ht = np.transpose(paf_ht, (1, 2, 0))
    if upscale is not None:
        paf_ht = cv2.resize(paf_ht, (paf_ht.shape[1] * upscale, paf_ht.shape[0] * upscale),
                            interpolation=cv2.INTER_CUBIC)

    if out_form == 3:
        limbSeq = [
            [[2, 0], [4, 1], [3, 0], [5, 1], [1, 0],
             [2, 2], [4, 4], [3, 3], [5, 5], [4, 2],
             [5, 3], [2, 3], [4, 5], [3, 2], [5, 4],
             [3, 5], [5, 5], [5, 5], [3, 3], [2, 2],
             [4, 4], [2, 6], [4, 7], [7, 6], [2, 8],
             [3, 8], [4, 9], [5, 9], [9, 8], [10, 0],
             [11, 0], [12, 1], [13, 1], [13, 11], [12, 10],
             [10, 11], [12, 13]]
        ]
        mapIdx = [
            [0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
            [10, 11], [12, 13], [14, 15], [16, 17], [18, 19],
            [20, 21], [22, 23], [24, 25], [26, 27], [28, 29],
            [30, 31], [32, 33], [34, 35], [36, 37], [38, 39],
            [40, 41], [42, 43], [44, 45], [46, 47], [48, 49],
            [50, 51], [52, 53], [54, 55], [56, 57], [58, 59],
            [60, 61], [62, 63], [64, 65], [66, 67], [68, 69],
            [70, 71], [72, 73]]

    else:
        limbSeq = [
            [1, 0], [1, 1], [1, 2], [1, 3], [1, 4]
        ]
        mapIdx = [
            [0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    all_connection = [[] for _ in mapIdx]
    mid_num = 10
    for k in range(len(mapIdx)):
        score_mid = paf_ht[:, :, [x for x in mapIdx[k]]]

        p1_index = limbSeq[k][0]
        p2_index = limbSeq[k][1]

        candA = all_peaks[p1_index]
        candB = all_peaks[p2_index]
        nA = len(candA)
        nB = len(candB)

        # indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            for i in range(nA):
                connection_candidate = None
                max_temp = 0
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    if norm == 0:
                        continue
                    norm = max(0.001, norm)
                    vec = np.divide(vec, norm)
                    #
                    # x, y = np.meshgrid(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                    #                    np.linspace(candA[i][1], candB[j][1], num=mid_num))
                    #
                    # vec_x = score_mid[y.round().astype(np.int32), x.round().astype(np.int32), 0]
                    # vec_y = score_mid[y.round().astype(np.int32), x.round().astype(np.int32), 1]

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in
                                      range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in
                                      range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * img_shape / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        # pt1_type,pt1_id,pt1_x,pt1_y
                        # pt2_type,pt2_id,pt2_x,pt2_y
                        # length, vec_x, vec_y,
                        # skeleton_type,skeleton_score, skeelton_and_pts_score
                        skeleton = [limbSeq[k][0] + 1, candA[i][2], candA[i][0], candA[i][1],
                                    limbSeq[k][1] + 1, candB[j][2], candB[j][0], candB[j][1],
                                    norm, vec[0], vec[1], k + 1, score_with_dist_prior,
                                    score_with_dist_prior + candA[i][2] + candB[j][2]]

                        if skeleton[-1] > max_temp:
                            connection_candidate = skeleton
                            max_temp = skeleton[-1]
                if connection_candidate is not None:
                    all_connection[k].append(connection_candidate)
    return all_connection


def draw_tower(all_connection, img, save_dir, all_peaks):
    saturation = 1.0  # 最大饱和度
    value = 1.0  # 最大明度
    bgr_colors = []
    hue_steps = 39  # 均分的色调数量
    for i in range(hue_steps):
        hue = i / hue_steps  # 计算色调
        rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)  # 转换为RGB颜色
        bgr_color = tuple([round(x * 255) for x in reversed(rgb_color)])  # 转换为BGR颜色
        bgr_colors.append(bgr_color)
    sk_colors = bgr_colors

    hue_steps = 14  # 均分的色调数量
    for i in range(hue_steps):
        hue = i / hue_steps  # 计算色调
        rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)  # 转换为RGB颜色
        bgr_color = tuple([round(x * 255) for x in reversed(rgb_color)])  # 转换为BGR颜色
        bgr_colors.append(bgr_color)
    pt_colors = bgr_colors

    pt_img = copy.deepcopy(img)
    pt_img = pt_img // 4
    for (all_peak, pt_color) in zip(all_peaks, pt_colors):
        for pt in all_peak:
            x = pt[0]
            y = pt[1]
            pt_img = cv2.circle(pt_img, center=(x, y), color=pt_color, radius=5, thickness=-1)

    save_path = os.path.join(save_dir, "0_pts.jpg")

    cv2.imwrite(save_path, pt_img)

    pt_drawn = []
    img = img // 4
    for all_connection_type in all_connection:
        for [pt1_type, pt1_id, pt1_x, pt1_y,
             pt2_type, pt2_id, pt2_x, pt2_y,
             length, vec_x, vec_y,
             skeleton_type, skeleton_score, skeelton_and_pts_score] in all_connection_type:
            if not pt1_id in pt_drawn:
                img = cv2.circle(img, center=(pt1_x, pt1_y), radius=5, color=pt_colors[pt1_type - 1],
                                 thickness=-1)
                pt_drawn.append(pt1_id)
            if not pt2_id in pt_drawn:
                img = cv2.circle(img, center=(pt2_x, pt2_y), radius=5, color=pt_colors[pt2_type - 1],
                                 thickness=-1)
                pt_drawn.append(pt2_id)
            img = cv2.line(img, (pt1_x, pt1_y), (pt2_x, pt2_y), thickness=2,
                           color=sk_colors[skeleton_type - 1])
    cv2.imwrite(os.path.join(save_dir, "0__show.jpg"), img)


def calculate_iou_like(kt1, kt2, sigma):
    [x1, y1, _, _] = kt1
    [x2, y2, _] = kt2

    dd = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

    oks = math.exp(-(dd / sigma / sigma))
    return oks

    # w1 = w2 = h1 = h2 = int(sigma * 3)
    #
    # left = max(x1, x2)
    # top = max(y1, y2)
    # right = min(x1 + w1, x2 + w2)
    # bottom = min(y1 + h1, y2 + h2)
    #
    # if right > left and bottom > top:
    #     intersection = (right - left) * (bottom - top)
    #     union = w1 * h1 + w2 * h2 - intersection
    #     return intersection / union
    # else:
    #     return 0.0


def calculate_precision_recall(detections, ground_truths, sigma, iou_threshold):
    tp = 0
    fp = 0
    fn = 0

    for detection in detections:
        max_iou = 0
        max_gt = None

        for ground_truth in ground_truths:
            iou = calculate_iou_like(detection, ground_truth, sigma)

            if iou > max_iou:
                max_iou = iou
                max_gt = ground_truth

        if max_iou >= iou_threshold:
            tp += 1
            ground_truths.remove(max_gt)
        else:
            fp += 1

    fn = len(ground_truths)

    return tp, fp, fn
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    #
    #
    # return precision, recall


def ljw_tower_pose_pack_accuracy(output: list,
                                 target: list,
                                 sigma: float,
                                 iou_threshold: float = 0.5) -> tuple:
    # N, K, H, W = output.shape

    tps = 0
    fps = 0
    fns = 0
    for (output_kt_type, target_kt_type) in zip(output, target):
        tp, fp, fn = calculate_precision_recall(output_kt_type, copy.deepcopy(target_kt_type), sigma,
                                                iou_threshold=iou_threshold)
        tps += tp
        fps += fp
        fns += fn
    return tps, fps, fns


def calculate_skeleton_precision_recall(detections, ground_truths, sigma, iou_threshold):
    tp = 0
    fp = 0
    fn = 0

    for detection in detections:
        max_iou = 0
        max_gt = None
        temp_iou_1 = 0
        temp_iou_2 = 0

        pr_p1 = [detection[2], detection[3], None, None]
        pr_p2 = [detection[6], detection[7], None, None]

        for ground_truth in ground_truths:
            gt_p1 = [ground_truth[0], ground_truth[1], None]
            gt_p2 = [ground_truth[3], ground_truth[4], None]

            iou_1 = calculate_iou_like(pr_p1, gt_p1, sigma)
            iou_2 = calculate_iou_like(pr_p2, gt_p2, sigma)

            if iou_1 + iou_2 > max_iou:
                max_iou = iou_1 + iou_2
                max_gt = ground_truth
                temp_iou_1 = iou_1
                temp_iou_2 = iou_2

        if temp_iou_1 >= iou_threshold and temp_iou_2 >= iou_threshold:
            tp += 1
            ground_truths.remove(max_gt)
        else:
            fp += 1

    fn = len(ground_truths)

    return tp, fp, fn


def ljw_tower_skeleton_accuracy(output: list,
                                target: list,
                                sigma: float,
                                iou_threshold: float = 0.5) -> tuple:
    tps = 0
    fps = 0
    fns = 0
    for (output_kt_type, target_kt_type) in zip(output, target):
        tp, fp, fn = calculate_skeleton_precision_recall(output_kt_type,
                                                         copy.deepcopy(target_kt_type),
                                                         sigma,
                                                         iou_threshold=iou_threshold)
        tps += tp
        fps += fp
        fns += fn
    return tps, fps, fns
