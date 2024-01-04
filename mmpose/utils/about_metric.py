# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Tuple
import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def get_all_peaks(heatmap, sigma):
    all_peaks = []
    all_scores = []
    peak_counter = 0
    thre1 = 0.1
    for part in range(heatmap.shape[0]):
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

        all_peaks.append(peaks_with_score_and_id)
        all_scores.append([peaks_with_score[i][-1] for i in range(len(peak_id))])
        peak_counter += len(peaks)
    return all_peaks, all_scores



def calculate_iou_like(kt1, kt2, sigma):
    [x1, y1, _, _] = kt1
    [x2, y2, _] = kt2

    w1 = w2 = h1 = h2 = int(sigma * 3)

    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)

    if right > left and bottom > top:
        intersection = (right - left) * (bottom - top)
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / union
    else:
        return 0.0


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
        tp, fp, fn = calculate_precision_recall(output_kt_type, copy.deepcopy(target_kt_type), sigma, iou_threshold=iou_threshold)
        tps += tp
        fps += fp
        fns += fn
    return tps, fps, fns