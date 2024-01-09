# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import cv2
import torch
import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils.gaussian_heatmap import (generate_gaussian_heatmaps,
                                     generate_unbiased_gaussian_heatmaps)
from .utils.post_processing import get_heatmap_maximum
from .utils.refinement import refine_keypoints, refine_keypoints_dark

from scipy.ndimage.filters import gaussian_filter
from mmpose.utils import get_all_peaks, get_all_skeletons


# def get_all_peaks(heatmap, sigma):
#     all_peaks = []
#     all_scores = []
#     peak_counter = 0
#     thre1 = 0.1
#     for part in range(heatmap.shape[0]):
#         map_ori = heatmap[part, :, :]
#         one_heatmap = gaussian_filter(map_ori, sigma=sigma)
#
#         map_left = np.zeros(one_heatmap.shape)
#         map_left[1:, :] = one_heatmap[:-1, :]
#         map_right = np.zeros(one_heatmap.shape)
#         map_right[:-1, :] = one_heatmap[1:, :]
#         map_up = np.zeros(one_heatmap.shape)
#         map_up[:, 1:] = one_heatmap[:, :-1]
#         map_down = np.zeros(one_heatmap.shape)
#         map_down[:, :-1] = one_heatmap[:, 1:]
#
#         peaks_binary = np.logical_and.reduce(
#             (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up,
#              one_heatmap >= map_down, one_heatmap > thre1))
#         peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
#         peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
#         peak_id = range(peak_counter, peak_counter + len(peaks))
#         peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]
#
#         all_peaks.append(peaks_with_score_and_id)
#         all_scores.append([peaks_with_score[i][-1] for i in range(len(peak_id))])
#         peak_counter += len(peaks)
#     return all_peaks, all_scores


def get_PAF(p1, p2, array, xx, yy, heatmap_scale, paf_half_width):
    p1 = np.array(p1) // heatmap_scale
    p2 = np.array(p2) // heatmap_scale
    length = np.sum((np.array(p1) - np.array(p2)) ** 2) ** 0.5
    e = (np.array(p2) - np.array(p1)) / length
    xs, ys = array[0], array[1]
    theta = np.arctan(np.clip((p2[1] - p1[1]) / (p2[0] - p1[0]), -1e8, 1e8))
    new_xs = (xx - p1[0]) * np.cos(theta) + (yy - p1[1]) * np.sin(theta)
    new_ys = -(xx - p1[0]) * np.sin(theta) + (yy - p1[1]) * np.cos(theta)

    p2_x = (p2[0] - p1[0]) * np.cos(theta) + (p2[1] - p1[1]) * np.sin(theta)

    if p2_x > 0:
        condition_1 = new_xs >= 0
        condition_2 = new_xs <= length
    else:
        condition_1 = new_xs <= 0
        condition_2 = new_xs >= -length
    condition_3 = new_ys <= paf_half_width
    condition_4 = new_ys >= - paf_half_width
    area = np.logical_and(condition_1, np.logical_and(condition_2, np.logical_and(condition_3, condition_4)))
    xs[area] = e[0]
    ys[area] = e[1]

    return np.array([xs, ys])


@KEYPOINT_CODECS.register_module()
class MSRAHeatmap(BaseKeypointCodec):
    """Represent keypoints as heatmaps via "MSRA" approach. See the paper:
    `Simple Baselines for Human Pose Estimation and Tracking`_ by Xiao et al
    (2018) for details.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]
        - heatmap size: [W, H]

    Encoded:

        - heatmaps (np.ndarray): The generated heatmap in shape (K, H, W)
            where [W, H] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape (N, K)

    Args:
        input_size (tuple): Image size in [w, h]
        heatmap_size (tuple): Heatmap size in [W, H]
        sigma (float): The sigma value of the Gaussian heatmap
        unbiased (bool): Whether use unbiased method (DarkPose) in ``'msra'``
            encoding. See `Dark Pose`_ for details. Defaults to ``False``
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation in DarkPose. The kernel size and sigma should follow
            the expirical formula :math:`sigma = 0.3*((ks-1)*0.5-1)+0.8`.
            Defaults to 11

    .. _`Simple Baselines for Human Pose Estimation and Tracking`:
        https://arxiv.org/abs/1804.06208
    .. _`Dark Pose`: https://arxiv.org/abs/1910.06278
    """

    label_mapping_table = dict(keypoint_weights='keypoint_weights', )
    field_mapping_table = dict(heatmaps='heatmaps', )

    def __init__(self,
                 input_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 sigma: float,
                 unbiased: bool = False,
                 blur_kernel_size: int = 11) -> None:
        super().__init__()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.unbiased = unbiased

        # The Gaussian blur kernel size of the heatmap modulation
        # in DarkPose and the sigma value follows the expirical
        # formula :math:`sigma = 0.3*((ks-1)*0.5-1)+0.8`
        # which gives:
        #   sigma~=3 if ks=17
        #   sigma=2 if ks=11;
        #   sigma~=1.5 if ks=7;
        #   sigma~=1 if ks=3;
        self.blur_kernel_size = blur_kernel_size
        self.scale_factor = (np.array(input_size) /
                             heatmap_size).astype(np.float32)

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Encode keypoints into heatmaps. Note that the original keypoint
        coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            dict:
            - heatmaps (np.ndarray): The generated heatmap in shape
                (K, H, W) where [W, H] is the `heatmap_size`
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """

        assert keypoints.shape[0] == 1, (
            f'{self.__class__.__name__} only support single-instance '
            'keypoint encoding')

        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        if self.unbiased:
            heatmaps, keypoint_weights = generate_unbiased_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                keypoints=keypoints / self.scale_factor,
                keypoints_visible=keypoints_visible,
                sigma=self.sigma)
        else:
            heatmaps, keypoint_weights = generate_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                keypoints=keypoints / self.scale_factor,
                keypoints_visible=keypoints_visible,
                sigma=self.sigma)

        encoded = dict(heatmaps=heatmaps, keypoint_weights=keypoint_weights)

        return encoded

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from heatmaps. The decoded keypoint
        coordinates are in the input image space.

        Args:
            encoded (np.ndarray): Heatmaps in shape (K, H, W)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded keypoint coordinates in shape
                (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K). It
                usually represents the confidence of the keypoint prediction
        """
        heatmaps = encoded.copy()
        K, H, W = heatmaps.shape

        keypoints, scores = get_heatmap_maximum(heatmaps)

        # Unsqueeze the instance dimension for single-instance results
        keypoints, scores = keypoints[None], scores[None]

        if self.unbiased:
            # Alleviate biased coordinate
            keypoints = refine_keypoints_dark(
                keypoints, heatmaps, blur_kernel_size=self.blur_kernel_size)

        else:
            keypoints = refine_keypoints(keypoints, heatmaps)

        # Restore the keypoint scale
        keypoints = keypoints * self.scale_factor

        return keypoints, scores


@KEYPOINT_CODECS.register_module()
class LJWHeatmapAndPaf(BaseKeypointCodec):
    label_mapping_table = dict(keypoint_weights='keypoint_weights', )
    field_mapping_table = dict(heatmaps='heatmaps', )

    def __init__(self,
                 input_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 sigma: float,
                 paf_half_width: int,
                 heatmap_scale: int,
                 num_keypoints: int,
                 num_skeletons: int,
                 unbiased: bool = False,
                 blur_kernel_size: int = 11) -> None:
        super().__init__()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.paf_half_width = paf_half_width
        self.unbiased = unbiased
        self.heatmap_scale = heatmap_scale
        self.num_keypoints = num_keypoints
        self.num_skeletons = num_skeletons
        # The Gaussian blur kernel size of the heatmap modulation
        # in DarkPose and the sigma value follows the expirical
        # formula :math:`sigma = 0.3*((ks-1)*0.5-1)+0.8`
        # which gives:
        #   sigma~=3 if ks=17
        #   sigma=2 if ks=11;
        #   sigma~=1.5 if ks=7;
        #   sigma~=1 if ks=3;
        self.blur_kernel_size = blur_kernel_size
        self.scale_factor = (np.array(input_size) /
                             heatmap_size).astype(np.float32)

    def encode(self, keypoints: list, skeletons: list) -> dict:
        """Encode keypoints into heatmaps. Note that the original keypoint
        coordinates should be in the input image space.
        """
        bgs = np.zeros((self.num_keypoints, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)

        paf_x = np.arange(self.heatmap_size[1])  # 生成包含横坐标值的一维数组，长度为 n
        paf_y = np.arange(self.heatmap_size[0])  # 生成包含纵坐标值的一维数组，长度为 m
        paf_bgs = np.zeros((self.num_skeletons * 2, self.heatmap_size[0], self.heatmap_size[1]), dtype=np.float32)

        xx, yy = np.meshgrid(paf_x, paf_y)
        xx += 1
        yy += 1
        for [x, y, keypoint_type] in keypoints:
            keypoint_type = keypoint_type
            left = max(0, x // self.heatmap_scale - 3 * self.sigma)
            right = min(self.heatmap_size[1], x // self.heatmap_scale + 3 * self.sigma)
            lower = max(0, y // self.heatmap_scale - 3 * self.sigma)
            upper = min(self.heatmap_size[0], y // self.heatmap_scale + 3 * self.sigma)
            heatmap = np.zeros([self.heatmap_size[0], self.heatmap_size[0]])
            heatmap[lower: upper, left:right] = np.exp(-((xx - x / self.heatmap_scale) ** 2 + (
                    yy - y / self.heatmap_scale) ** 2) / 2 / self.sigma / self.sigma)[lower: upper, left:right]
            bgs[keypoint_type - 1][lower: upper, left:right] += heatmap[lower: upper, left:right]

        for [p1x, p1y, p1t, p2x, p2y, p2t, skeleton_type] in skeletons:
            paf_index = (skeleton_type - 1) * 2
            p1x = int(p1x)
            p1y = int(p1y)
            p1t = int(p1t)
            p2x = int(p2x)
            p2y = int(p2y)
            p2t = int(p2t)

            paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y), paf_bgs[paf_index:paf_index + 2],
                                                       xx, yy, self.heatmap_scale, self.paf_half_width)

            # if skeleton_type == 2:
            #     if p1y < p2y:
            #         paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y), paf_bgs[paf_index:paf_index + 2],
            #                                                    xx, yy, self.heatmap_scale, self.paf_half_width)
            #     else:
            #         paf_bgs[paf_index:paf_index + 2] = get_PAF((p2x, p2y), (p1x, p1y), paf_bgs[paf_index:paf_index + 2],
            #                                                    xx, yy, self.heatmap_scale, self.paf_half_width)
            #
            # elif skeleton_type == 5:
            #     if p1t == 5:
            #         paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y), paf_bgs[paf_index:paf_index + 2],
            #                                                    xx, yy, self.heatmap_scale, self.paf_half_width)
            #     else:
            #         paf_bgs[paf_index:paf_index + 2] = get_PAF((p2x, p2y), (p1x, p1y), paf_bgs[paf_index:paf_index + 2],
            #                                                    xx, yy, self.heatmap_scale, self.paf_half_width)
            # else:
            #     if p1t == 2:
            #         paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y), paf_bgs[paf_index:paf_index + 2],
            #                                                    xx, yy, self.heatmap_scale, self.paf_half_width)
            #     else:
            #         paf_bgs[paf_index:paf_index + 2] = get_PAF((p2x, p2y), (p1x, p1y), paf_bgs[paf_index:paf_index + 2],
            #                                                    xx, yy, self.heatmap_scale, self.paf_half_width)
        heatmaps = np.vstack([bgs, paf_bgs])

        encoded = dict(heatmaps=heatmaps, keypoint_weights=np.ones((1, heatmaps.shape[0]), dtype=np.float32))

        return encoded

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from heatmaps. The decoded keypoint
        coordinates are in the input image space.

        Args:
            encoded (np.ndarray): Heatmaps in shape (K, H, W)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded keypoint coordinates in shape
                (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K). It
                usually represents the confidence of the keypoint prediction
        """
        heatmaps = encoded.copy()
        K, H, W = heatmaps.shape

        keypoints, scores = get_heatmap_maximum(heatmaps)

        # Unsqueeze the instance dimension for single-instance results
        keypoints, scores = keypoints[None], scores[None]

        if self.unbiased:
            # Alleviate biased coordinate
            keypoints = refine_keypoints_dark(
                keypoints, heatmaps, blur_kernel_size=self.blur_kernel_size)

        else:
            keypoints = refine_keypoints(keypoints, heatmaps)

        # Restore the keypoint scale
        keypoints = keypoints * self.scale_factor

        return keypoints, scores


@KEYPOINT_CODECS.register_module()
class LJWHeatmapAndPaf_2_(BaseKeypointCodec):
    label_mapping_table = dict(keypoint_weights='keypoint_weights', )
    field_mapping_table = dict(heatmaps='heatmaps', )

    def __init__(self,
                 input_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 heatmap_scale: int,
                 output_form_1: bool,
                 sigma_1: float,
                 paf_half_width_1: int,
                 num_keypoints_1: int,
                 num_skeletons_1: int,
                 output_form_2: bool,
                 sigma_2: float,
                 paf_half_width_2: int,
                 num_keypoints_2: int,
                 num_skeletons_2: int,
                 output_form_3: bool,
                 sigma_3: float,
                 paf_half_width_3: int,
                 num_keypoints_3: int,
                 num_skeletons_3: int,
                 use_medium_satge: bool = True,
                 target_form: int = 3,
                 unbiased: bool = False,
                 blur_kernel_size: int = 11) -> None:
        super().__init__()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.use_medium_satge = use_medium_satge
        self.target_form = target_form
        self.output_form_1 = output_form_1
        self.sigma_1 = sigma_1
        self.paf_half_width_1 = paf_half_width_1
        self.num_keypoints_1 = num_keypoints_1
        self.num_skeletons_1 = num_skeletons_1
        self.output_form_2 = output_form_2
        self.sigma_2 = sigma_2
        self.paf_half_width_2 = paf_half_width_2
        self.num_keypoints_2 = num_keypoints_2
        self.num_skeletons_2 = num_skeletons_2
        self.output_form_3 = output_form_3
        self.sigma_3 = sigma_3
        self.paf_half_width_3 = paf_half_width_3
        self.num_keypoints_3 = num_keypoints_3
        self.num_skeletons_3 = num_skeletons_3
        self.unbiased = unbiased
        self.heatmap_scale = heatmap_scale
        # The Gaussian blur kernel size of the heatmap modulation
        # in DarkPose and the sigma value follows the expirical
        # formula :math:`sigma = 0.3*((ks-1)*0.5-1)+0.8`
        # which gives:
        #   sigma~=3 if ks=17
        #   sigma=2 if ks=11;
        #   sigma~=1.5 if ks=7;
        #   sigma~=1 if ks=3;
        self.blur_kernel_size = blur_kernel_size
        self.scale_factor = (np.array(input_size) /
                             heatmap_size).astype(np.float32)

    def encode(self,
               keypoints_1: list = [],
               skeletons_1: list = [],
               keypoints_2: list = [],
               skeletons_2: list = [],
               keypoints_3: list = [],
               skeletons_3: list = [], ) -> dict:
        """Encode keypoints into heatmaps. Note that the original keypoint
        coordinates should be in the input image space.
        """

        if self.use_medium_satge:
            lst = [None] * 12

            if self.output_form_1:
                bgs = np.zeros((self.num_keypoints_1, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)

                paf_x = np.arange(self.heatmap_size[1])  # 生成包含横坐标值的一维数组，长度为 n
                paf_y = np.arange(self.heatmap_size[0])  # 生成包含纵坐标值的一维数组，长度为 m
                paf_bgs = np.zeros((self.num_skeletons_1 * 2, self.heatmap_size[0], self.heatmap_size[1]),
                                   dtype=np.float32)

                xx, yy = np.meshgrid(paf_x, paf_y)
                xx += 1
                yy += 1
                for [x, y, keypoint_type] in keypoints_1:
                    keypoint_type = keypoint_type
                    left = max(0, x // self.heatmap_scale - 3 * self.sigma_1)
                    right = min(self.heatmap_size[1], x // self.heatmap_scale + 3 * self.sigma_1)
                    lower = max(0, y // self.heatmap_scale - 3 * self.sigma_1)
                    upper = min(self.heatmap_size[0], y // self.heatmap_scale + 3 * self.sigma_1)
                    heatmap = np.zeros([self.heatmap_size[0], self.heatmap_size[0]])
                    heatmap[lower: upper, left:right] = np.exp(-((xx - x / self.heatmap_scale) ** 2 + (
                            yy - y / self.heatmap_scale) ** 2) / 2 / self.sigma_1 / self.sigma_1)[lower: upper,
                                                        left:right]
                    bgs[keypoint_type - 1][lower: upper, left:right] += heatmap[lower: upper, left:right]

                for [p1x, p1y, p1t, p2x, p2y, p2t, skeleton_type] in skeletons_1:
                    paf_index = (skeleton_type - 1) * 2
                    p1x = int(p1x)
                    p1y = int(p1y)
                    p1t = int(p1t)
                    p2x = int(p2x)
                    p2y = int(p2y)
                    p2t = int(p2t)

                    paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y), paf_bgs[paf_index:paf_index + 2],
                                                               xx, yy, self.heatmap_scale, self.paf_half_width_1)

                    # if skeleton_type == 2:
                    #     if p1y < p2y:
                    #         paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y), paf_bgs[paf_index:paf_index + 2],
                    #                                                    xx, yy, self.heatmap_scale, self.paf_half_width)
                    #     else:
                    #         paf_bgs[paf_index:paf_index + 2] = get_PAF((p2x, p2y), (p1x, p1y), paf_bgs[paf_index:paf_index + 2],
                    #                                                    xx, yy, self.heatmap_scale, self.paf_half_width)
                    #
                    # elif skeleton_type == 5:
                    #     if p1t == 5:
                    #         paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y), paf_bgs[paf_index:paf_index + 2],
                    #                                                    xx, yy, self.heatmap_scale, self.paf_half_width)
                    #     else:
                    #         paf_bgs[paf_index:paf_index + 2] = get_PAF((p2x, p2y), (p1x, p1y), paf_bgs[paf_index:paf_index + 2],
                    #                                                    xx, yy, self.heatmap_scale, self.paf_half_width)
                    # else:
                    #     if p1t == 2:
                    #         paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y), paf_bgs[paf_index:paf_index + 2],
                    #                                                    xx, yy, self.heatmap_scale, self.paf_half_width)
                    #     else:
                    #         paf_bgs[paf_index:paf_index + 2] = get_PAF((p2x, p2y), (p1x, p1y), paf_bgs[paf_index:paf_index + 2],
                    #                                                    xx, yy, self.heatmap_scale, self.paf_half_width)

                lst[0], lst[2], lst[6] = bgs, bgs, bgs
                lst[1], lst[3], lst[7] = paf_bgs, paf_bgs, paf_bgs
            if self.output_form_2:
                bgs = np.zeros((self.num_keypoints_2, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)

                paf_x = np.arange(self.heatmap_size[1])  # 生成包含横坐标值的一维数组，长度为 n
                paf_y = np.arange(self.heatmap_size[0])  # 生成包含纵坐标值的一维数组，长度为 m
                paf_bgs = np.zeros((self.num_skeletons_2 * 2, self.heatmap_size[0], self.heatmap_size[1]),
                                   dtype=np.float32)

                xx, yy = np.meshgrid(paf_x, paf_y)
                xx += 1
                yy += 1
                for [x, y, keypoint_type] in keypoints_2:
                    keypoint_type = keypoint_type
                    left = int(max(0, x // self.heatmap_scale - 3 * self.sigma_2))
                    right = int(min(self.heatmap_size[1], x // self.heatmap_scale + 3 * self.sigma_2))
                    lower = int(max(0, y // self.heatmap_scale - 3 * self.sigma_2))
                    upper = int(min(self.heatmap_size[0], y // self.heatmap_scale + 3 * self.sigma_2))
                    heatmap = np.zeros([self.heatmap_size[0], self.heatmap_size[0]])
                    heatmap[lower: upper, left:right] = np.exp(-((xx - x / self.heatmap_scale) ** 2 + (
                            yy - y / self.heatmap_scale) ** 2) / 2 / self.sigma_2 / self.sigma_2)[lower: upper,
                                                        left:right]
                    bgs[keypoint_type - 1][lower: upper, left:right] += heatmap[lower: upper, left:right]

                for [p1x, p1y, p1t, p2x, p2y, p2t, skeleton_type] in skeletons_2:
                    paf_index = (skeleton_type - 1) * 2
                    p1x = int(p1x)
                    p1y = int(p1y)
                    p1t = int(p1t)
                    p2x = int(p2x)
                    p2y = int(p2y)
                    p2t = int(p2t)

                    paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y), paf_bgs[paf_index:paf_index + 2],
                                                               xx, yy, self.heatmap_scale, self.paf_half_width_2)

                lst[4], lst[8] = bgs, bgs
                lst[5], lst[9] = paf_bgs, paf_bgs
            if self.output_form_3:
                bgs = np.zeros((self.num_keypoints_3, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)

                paf_x = np.arange(self.heatmap_size[1])  # 生成包含横坐标值的一维数组，长度为 n
                paf_y = np.arange(self.heatmap_size[0])  # 生成包含纵坐标值的一维数组，长度为 m
                paf_bgs = np.zeros((self.num_skeletons_3 * 2, self.heatmap_size[0], self.heatmap_size[1]),
                                   dtype=np.float32)

                xx, yy = np.meshgrid(paf_x, paf_y)
                xx += 1
                yy += 1
                for [x, y, keypoint_type] in keypoints_3:
                    keypoint_type = keypoint_type
                    left = int(max(0, x // self.heatmap_scale - 3 * self.sigma_3))
                    right = int(min(self.heatmap_size[1], x // self.heatmap_scale + 3 * self.sigma_3))
                    lower = int(max(0, y // self.heatmap_scale - 3 * self.sigma_3))
                    upper = int(min(self.heatmap_size[0], y // self.heatmap_scale + 3 * self.sigma_3))
                    heatmap = np.zeros([self.heatmap_size[0], self.heatmap_size[0]])
                    heatmap[lower: upper, left:right] = np.exp(-((xx - x / self.heatmap_scale) ** 2 + (
                            yy - y / self.heatmap_scale) ** 2) / 2 / self.sigma_3 / self.sigma_3)[lower: upper,
                                                        left:right]
                    bgs[keypoint_type - 1][lower: upper, left:right] += heatmap[lower: upper, left:right]

                for [p1x, p1y, p1t, p2x, p2y, p2t, skeleton_type] in skeletons_3:
                    paf_index = (skeleton_type - 1) * 2
                    p1x = int(p1x)
                    p1y = int(p1y)
                    p1t = int(p1t)
                    p2x = int(p2x)
                    p2y = int(p2y)
                    p2t = int(p2t)

                    if skeleton_type in [6, 7, 8, 9, 18, 19, 20, 21]:
                        if p2x > p1x:
                            paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y),
                                                                       paf_bgs[paf_index:paf_index + 2],
                                                                       xx, yy, self.heatmap_scale,
                                                                       self.paf_half_width_3)
                        else:
                            paf_bgs[paf_index:paf_index + 2] = get_PAF((p2x, p2y), (p1x, p1y),
                                                                       paf_bgs[paf_index:paf_index + 2],
                                                                       xx, yy, self.heatmap_scale,
                                                                       self.paf_half_width_3)
                    else:
                        paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y),
                                                                   paf_bgs[paf_index:paf_index + 2],
                                                                   xx, yy, self.heatmap_scale, self.paf_half_width_3)

                lst[10] = bgs
                lst[11] = paf_bgs
            lst = [x for x in lst if x is not None]
        else:
            lst = [None] * 2
            if self.target_form == 1:
                bgs = np.zeros((self.num_keypoints_1, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)

                paf_x = np.arange(self.heatmap_size[1])  # 生成包含横坐标值的一维数组，长度为 n
                paf_y = np.arange(self.heatmap_size[0])  # 生成包含纵坐标值的一维数组，长度为 m
                paf_bgs = np.zeros((self.num_skeletons_1 * 2, self.heatmap_size[0], self.heatmap_size[1]),
                                   dtype=np.float32)

                xx, yy = np.meshgrid(paf_x, paf_y)
                xx += 1
                yy += 1
                for [x, y, keypoint_type] in keypoints_1:
                    keypoint_type = keypoint_type
                    left = int(max(0, x // self.heatmap_scale - 3 * self.sigma_1))
                    right = int(min(self.heatmap_size[1], x // self.heatmap_scale + 3 * self.sigma_1))
                    lower = int(max(0, y // self.heatmap_scale - 3 * self.sigma_1))
                    upper = int(min(self.heatmap_size[0], y // self.heatmap_scale + 3 * self.sigma_1))
                    heatmap = np.zeros([self.heatmap_size[0], self.heatmap_size[0]])
                    heatmap[lower: upper, left:right] = np.exp(-((xx - x / self.heatmap_scale) ** 2 + (
                            yy - y / self.heatmap_scale) ** 2) / 2 / self.sigma_1 / self.sigma_1)[lower: upper,
                                                        left:right]
                    bgs[keypoint_type - 1][lower: upper, left:right] += heatmap[lower: upper, left:right]

                for [p1x, p1y, p1t, p2x, p2y, p2t, skeleton_type] in skeletons_1:
                    paf_index = (skeleton_type - 1) * 2
                    p1x = int(p1x)
                    p1y = int(p1y)
                    p1t = int(p1t)
                    p2x = int(p2x)
                    p2y = int(p2y)
                    p2t = int(p2t)

                    paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y), paf_bgs[paf_index:paf_index + 2],
                                                               xx, yy, self.heatmap_scale, self.paf_half_width_1)

                lst[0] = bgs
                lst[1] = paf_bgs
            if self.target_form == 2:
                bgs = np.zeros((self.num_keypoints_2, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)

                paf_x = np.arange(self.heatmap_size[1])  # 生成包含横坐标值的一维数组，长度为 n
                paf_y = np.arange(self.heatmap_size[0])  # 生成包含纵坐标值的一维数组，长度为 m
                paf_bgs = np.zeros((self.num_skeletons_2 * 2, self.heatmap_size[0], self.heatmap_size[1]),
                                   dtype=np.float32)

                xx, yy = np.meshgrid(paf_x, paf_y)
                xx += 1
                yy += 1
                for [x, y, keypoint_type] in keypoints_2:
                    keypoint_type = keypoint_type
                    left = int(max(0, x // self.heatmap_scale - 3 * self.sigma_2))
                    right = int(min(self.heatmap_size[1], x // self.heatmap_scale + 3 * self.sigma_2))
                    lower = int(max(0, y // self.heatmap_scale - 3 * self.sigma_2))
                    upper = int(min(self.heatmap_size[0], y // self.heatmap_scale + 3 * self.sigma_2))
                    heatmap = np.zeros([self.heatmap_size[0], self.heatmap_size[0]])
                    heatmap[lower: upper, left:right] = np.exp(-((xx - x / self.heatmap_scale) ** 2 + (
                            yy - y / self.heatmap_scale) ** 2) / 2 / self.sigma_2 / self.sigma_2)[lower: upper,
                                                        left:right]
                    bgs[keypoint_type - 1][lower: upper, left:right] += heatmap[lower: upper, left:right]

                for [p1x, p1y, p1t, p2x, p2y, p2t, skeleton_type] in skeletons_2:
                    paf_index = (skeleton_type - 1) * 2
                    p1x = int(p1x)
                    p1y = int(p1y)
                    p1t = int(p1t)
                    p2x = int(p2x)
                    p2y = int(p2y)
                    p2t = int(p2t)

                    paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y), paf_bgs[paf_index:paf_index + 2],
                                                               xx, yy, self.heatmap_scale, self.paf_half_width_2)

                lst[0] = bgs
                lst[1] = paf_bgs
            if self.target_form == 3:
                bgs = np.zeros((self.num_keypoints_3, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)

                paf_x = np.arange(self.heatmap_size[1])  # 生成包含横坐标值的一维数组，长度为 n
                paf_y = np.arange(self.heatmap_size[0])  # 生成包含纵坐标值的一维数组，长度为 m
                paf_bgs = np.zeros((self.num_skeletons_3 * 2, self.heatmap_size[0], self.heatmap_size[1]),
                                   dtype=np.float32)

                xx, yy = np.meshgrid(paf_x, paf_y)
                xx += 1
                yy += 1
                for [x, y, keypoint_type] in keypoints_3:
                    keypoint_type = keypoint_type
                    left = int(max(0, x // self.heatmap_scale - 3 * self.sigma_3))
                    right = int(min(self.heatmap_size[1], x // self.heatmap_scale + 3 * self.sigma_3))
                    lower = int(max(0, y // self.heatmap_scale - 3 * self.sigma_3))
                    upper = int(min(self.heatmap_size[0], y // self.heatmap_scale + 3 * self.sigma_3))
                    heatmap = np.zeros([self.heatmap_size[0], self.heatmap_size[0]])
                    heatmap[lower: upper, left:right] = np.exp(-((xx - x / self.heatmap_scale) ** 2 + (
                            yy - y / self.heatmap_scale) ** 2) / 2 / self.sigma_3 / self.sigma_3)[lower: upper,
                                                        left:right]
                    bgs[keypoint_type - 1][lower: upper, left:right] += heatmap[lower: upper, left:right]

                for [p1x, p1y, p1t, p2x, p2y, p2t, skeleton_type] in skeletons_3:
                    paf_index = (skeleton_type - 1) * 2
                    p1x = int(p1x)
                    p1y = int(p1y)
                    p1t = int(p1t)
                    p2x = int(p2x)
                    p2y = int(p2y)
                    p2t = int(p2t)

                    if skeleton_type in [6, 7, 8, 9, 18, 19, 20, 21]:
                        if p2x > p1x:
                            paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y),
                                                                       paf_bgs[paf_index:paf_index + 2],
                                                                       xx, yy, self.heatmap_scale,
                                                                       self.paf_half_width_3)
                        else:
                            paf_bgs[paf_index:paf_index + 2] = get_PAF((p2x, p2y), (p1x, p1y),
                                                                       paf_bgs[paf_index:paf_index + 2],
                                                                       xx, yy, self.heatmap_scale,
                                                                       self.paf_half_width_3)
                    else:
                        paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y),
                                                                   paf_bgs[paf_index:paf_index + 2],
                                                                   xx, yy, self.heatmap_scale, self.paf_half_width_3)

                lst[0] = bgs
                lst[1] = paf_bgs

        heatmaps = np.vstack(lst)

        keypoint_weights = np.ones((1, heatmaps.shape[0]), dtype=np.float32)
        keypoint_weights[:, -49] *= 2
        keypoint_weights[:, -47] *= 2
        keypoint_weights[:, -46] *= 2
        keypoint_weights[:, -45] *= 2
        encoded = dict(heatmaps=heatmaps, keypoint_weights=keypoint_weights)

        return encoded

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from heatmaps. The decoded keypoint
        coordinates are in the input image space.

        Args:
            encoded (np.ndarray): Heatmaps in shape (K, H, W)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded keypoint coordinates in shape
                (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K). It
                usually represents the confidence of the keypoint prediction
        """
        heatmaps = encoded.copy()

        assert self.output_form_3 == True

        pred_pt = heatmaps[-(self.num_keypoints_3 + self.num_skeletons_3 * 2):-self.num_skeletons_3 * 2, :, :]

        all_peaks, all_scroes = get_all_peaks(pred_pt, sigma=self.sigma_3)

        return all_peaks, all_scroes

        # keypoints, scores = get_heatmap_maximum(heatmaps)
        #
        # # Unsqueeze the instance dimension for single-instance results
        # keypoints, scores = keypoints[None], scores[None]
        #
        # if self.unbiased:
        #     # Alleviate biased coordinate
        #     keypoints = refine_keypoints_dark(
        #         keypoints, heatmaps, blur_kernel_size=self.blur_kernel_size)
        #
        # else:
        #     keypoints = refine_keypoints(keypoints, heatmaps)
        #
        # # Restore the keypoint scale
        # keypoints = keypoints * self.scale_factor
        #
        # return keypoints, scores


@KEYPOINT_CODECS.register_module()
class LJWHeatmapAndPaf_2(BaseKeypointCodec):
    label_mapping_table = dict(keypoint_weights='keypoint_weights', )
    field_mapping_table = dict(heatmaps='heatmaps', )

    def __init__(self,
                 input_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 heatmap_scale: int,
                 output_form_1: bool,
                 sigma_1: float,
                 paf_half_width_1: int,
                 num_keypoints_1: int,
                 num_skeletons_1: int,
                 output_form_2: bool,
                 sigma_2: float,
                 paf_half_width_2: int,
                 num_keypoints_2: int,
                 num_skeletons_2: int,
                 output_form_3: bool,
                 sigma_3: float,
                 paf_half_width_3: int,
                 num_keypoints_3: int,
                 num_skeletons_3: int,
                 use_medium_satge: bool = True,
                 target_form: int = 3,
                 unbiased: bool = False,
                 blur_kernel_size: int = 11) -> None:
        super().__init__()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.use_medium_satge = use_medium_satge
        self.target_form = target_form
        self.output_form_1 = output_form_1
        self.sigma_1 = sigma_1
        self.paf_half_width_1 = paf_half_width_1
        self.num_keypoints_1 = num_keypoints_1
        self.num_skeletons_1 = num_skeletons_1
        self.output_form_2 = output_form_2
        self.sigma_2 = sigma_2
        self.paf_half_width_2 = paf_half_width_2
        self.num_keypoints_2 = num_keypoints_2
        self.num_skeletons_2 = num_skeletons_2
        self.output_form_3 = output_form_3
        self.sigma_3 = sigma_3
        self.paf_half_width_3 = paf_half_width_3
        self.num_keypoints_3 = num_keypoints_3
        self.num_skeletons_3 = num_skeletons_3
        self.unbiased = unbiased
        self.heatmap_scale = heatmap_scale
        # The Gaussian blur kernel size of the heatmap modulation
        # in DarkPose and the sigma value follows the expirical
        # formula :math:`sigma = 0.3*((ks-1)*0.5-1)+0.8`
        # which gives:
        #   sigma~=3 if ks=17
        #   sigma=2 if ks=11;
        #   sigma~=1.5 if ks=7;
        #   sigma~=1 if ks=3;
        self.blur_kernel_size = blur_kernel_size
        self.scale_factor = (np.array(input_size) /
                             heatmap_size[-2:]).astype(np.float32)

    def encode(self,
               keypoints_1: list = [],
               skeletons_1: list = [],
               keypoints_2: list = [],
               skeletons_2: list = [],
               keypoints_3: list = [],
               skeletons_3: list = [], ) -> dict:
        """Encode keypoints into heatmaps. Note that the original keypoint
        coordinates should be in the input image space.
        """

        # ----------------------------------------------------------------------------------------------------------------
        # if self.output_form_1:
        # ----------------------------------------------------------------------------------------------------------------
        bgs = np.zeros((self.num_keypoints_1, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)

        paf_x = np.arange(self.heatmap_size[1])  # 生成包含横坐标值的一维数组，长度为 n
        paf_y = np.arange(self.heatmap_size[0])  # 生成包含纵坐标值的一维数组，长度为 m
        paf_bgs = np.zeros((self.num_skeletons_1 * 2, self.heatmap_size[1], self.heatmap_size[0]),
                           dtype=np.float32)
        xx, yy = np.meshgrid(paf_x, paf_y)
        xx += 1
        yy += 1
        for [x, y, keypoint_type] in keypoints_1:
            keypoint_type = keypoint_type
            left = int(max(0, x // self.heatmap_scale[0] - 3 * self.sigma_1))
            right = int(min(self.heatmap_size[1], x // self.heatmap_scale[0] + 3 * self.sigma_1))
            lower = int(max(0, y // self.heatmap_scale[0] - 3 * self.sigma_1))
            upper = int(min(self.heatmap_size[1], y // self.heatmap_scale[0] + 3 * self.sigma_1))
            heatmap = np.zeros([self.heatmap_size[1], self.heatmap_size[0]])
            heatmap[lower: upper, left:right] = np.exp(-((xx - x / self.heatmap_scale[0]) ** 2 + (
                    yy - y / self.heatmap_scale[0]) ** 2) / 2 / self.sigma_1 / self.sigma_1)[lower: upper,
                                                left:right]
            bgs[keypoint_type - 1][lower: upper, left:right] += heatmap[lower: upper, left:right]

        for [p1x, p1y, p1t, p2x, p2y, p2t, skeleton_type] in skeletons_1:
            paf_index = (skeleton_type - 1) * 2
            p1x = int(p1x)
            p1y = int(p1y)
            p1t = int(p1t)
            p2x = int(p2x)
            p2y = int(p2y)
            p2t = int(p2t)

            paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y), paf_bgs[paf_index:paf_index + 2],
                                                       xx, yy, self.heatmap_scale[0], self.paf_half_width_1)

        ht1 = np.vstack([bgs, paf_bgs])
        # ----------------------------------------------------------------------------------------------------------------
        # if self.output_form_2:
        # ----------------------------------------------------------------------------------------------------------------
        bgs = np.zeros((self.num_keypoints_2, self.heatmap_size[3], self.heatmap_size[2]), dtype=np.float32)

        paf_x = np.arange(self.heatmap_size[3])  # 生成包含横坐标值的一维数组，长度为 n
        paf_y = np.arange(self.heatmap_size[2])  # 生成包含纵坐标值的一维数组，长度为 m
        paf_bgs = np.zeros((self.num_skeletons_2 * 2, self.heatmap_size[3], self.heatmap_size[2]),
                           dtype=np.float32)

        xx, yy = np.meshgrid(paf_x, paf_y)
        xx += 1
        yy += 1
        for [x, y, keypoint_type] in keypoints_2:
            keypoint_type = keypoint_type
            left = int(max(0, x // self.heatmap_scale[1] - 3 * self.sigma_2))
            right = int(min(self.heatmap_size[3], x // self.heatmap_scale[1] + 3 * self.sigma_2))
            lower = int(max(0, y // self.heatmap_scale[1] - 3 * self.sigma_2))
            upper = int(min(self.heatmap_size[2], y // self.heatmap_scale[1] + 3 * self.sigma_2))
            heatmap = np.zeros([self.heatmap_size[3], self.heatmap_size[2]])
            heatmap[lower: upper, left:right] = np.exp(-((xx - x / self.heatmap_scale[1]) ** 2 + (
                    yy - y / self.heatmap_scale[1]) ** 2) / 2 / self.sigma_2 / self.sigma_2)[lower: upper,
                                                left:right]
            bgs[keypoint_type - 1][lower: upper, left:right] += heatmap[lower: upper, left:right]

        # for bg_idx, bg in enumerate(bgs):
        #     bg *= 255
        #     bg = bg.astype(np.uint8)
        #     cv2.imwrite(r"E:\LJW\Git\mmpose\tools\0_LJW_tools\show\_{}.jpg".format(bg_idx), img=bg)

        for [p1x, p1y, p1t, p2x, p2y, p2t, skeleton_type] in skeletons_2:
            paf_index = (skeleton_type - 1) * 2
            p1x = int(p1x)
            p1y = int(p1y)
            p1t = int(p1t)
            p2x = int(p2x)
            p2y = int(p2y)
            p2t = int(p2t)

            paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y), paf_bgs[paf_index:paf_index + 2],
                                                       xx, yy, self.heatmap_scale[1], self.paf_half_width_2)

        ht2 = np.vstack([bgs, paf_bgs])
        # ----------------------------------------------------------------------------------------------------------------
        # if self.output_form_3:
        # ----------------------------------------------------------------------------------------------------------------
        bgs = np.zeros((self.num_keypoints_3, self.heatmap_size[5], self.heatmap_size[4]), dtype=np.float32)

        paf_x = np.arange(self.heatmap_size[5])  # 生成包含横坐标值的一维数组，长度为 n
        paf_y = np.arange(self.heatmap_size[4])  # 生成包含纵坐标值的一维数组，长度为 m
        paf_bgs = np.zeros((self.num_skeletons_3 * 2, self.heatmap_size[5], self.heatmap_size[4]),
                           dtype=np.float32)

        xx, yy = np.meshgrid(paf_x, paf_y)
        xx += 1
        yy += 1
        for [x, y, keypoint_type] in keypoints_3:
            keypoint_type = keypoint_type
            left = int(max(0, x // self.heatmap_scale[2] - 3 * self.sigma_3))
            right = int(min(self.heatmap_size[5], x // self.heatmap_scale[2] + 3 * self.sigma_3))
            lower = int(max(0, y // self.heatmap_scale[2] - 3 * self.sigma_3))
            upper = int(min(self.heatmap_size[4], y // self.heatmap_scale[2] + 3 * self.sigma_3))
            heatmap = np.zeros([self.heatmap_size[5], self.heatmap_size[4]])
            heatmap[lower: upper, left:right] = np.exp(-((xx - x / self.heatmap_scale[2]) ** 2 + (
                    yy - y / self.heatmap_scale[2]) ** 2) / 2 / self.sigma_3 / self.sigma_3)[lower: upper,
                                                left:right]
            bgs[keypoint_type - 1][lower: upper, left:right] += heatmap[lower: upper, left:right]

        for [p1x, p1y, p1t, p2x, p2y, p2t, skeleton_type] in skeletons_3:
            paf_index = (skeleton_type - 1) * 2
            p1x = int(p1x)
            p1y = int(p1y)
            p1t = int(p1t)
            p2x = int(p2x)
            p2y = int(p2y)
            p2t = int(p2t)

            if skeleton_type in [6, 7, 8, 9, 18, 19, 20, 21]:
                if p2x > p1x:
                    paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y),
                                                               paf_bgs[paf_index:paf_index + 2],
                                                               xx, yy, self.heatmap_scale[2],
                                                               self.paf_half_width_3)
                else:
                    paf_bgs[paf_index:paf_index + 2] = get_PAF((p2x, p2y), (p1x, p1y),
                                                               paf_bgs[paf_index:paf_index + 2],
                                                               xx, yy, self.heatmap_scale[2],
                                                               self.paf_half_width_3)
            else:
                paf_bgs[paf_index:paf_index + 2] = get_PAF((p1x, p1y), (p2x, p2y),
                                                           paf_bgs[paf_index:paf_index + 2],
                                                           xx, yy, self.heatmap_scale[2], self.paf_half_width_3)

        ht3 = np.vstack([bgs, paf_bgs])
        # ----------------------------------------------------------------------------------------------------------------
        encoded = dict(
            heatmaps=ht3,
            keypoint_weights=np.ones((1, ht3.shape[0]), dtype=np.float32),
            heatmaps_1=torch.from_numpy(ht1),
            keypoint_weights_1=torch.from_numpy(np.ones((1, ht1.shape[0]), dtype=np.float32)),
            heatmaps_2=torch.from_numpy(ht2),
            keypoint_weights_2=torch.from_numpy(np.ones((1, ht2.shape[0]), dtype=np.float32)),
            heatmaps_3=torch.from_numpy(ht3),
            keypoint_weights_3=torch.from_numpy(np.ones((1, ht3.shape[0]), dtype=np.float32)),
        )

        return encoded

    def decode(self, ht1, ht2, ht3):
        """Decode keypoint coordinates from heatmaps. The decoded keypoint
        coordinates are in the input image space.

        Args:
            encoded (np.ndarray): Heatmaps in shape (K, H, W)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded keypoint coordinates in shape
                (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K). It
                usually represents the confidence of the keypoint prediction
        """
        # heatmaps = encoded.copy()

        assert self.output_form_3 == True

        img_shape = ht1.shape[2] * 16

        heatmap_kpt_1 = ht1[0, :self.num_keypoints_1, :, :]
        all_peaks_1, all_scroes_1 = get_all_peaks(heatmap_kpt_1, sigma=self.sigma_1, idx_channel=0, upscale=16)

        # heatmap_paf_1 = ht1[0, self.num_keypoints_1:, :, :]
        # heatmap_paf_1 = np.transpose(heatmap_paf_1, (1, 2, 0))
        # heatmap_paf_1 = cv2.resize(heatmap_paf_1, (img_shape, img_shape), interpolation=cv2.INTER_CUBIC)
        # all_skeletons_1 = get_all_skeletons(heatmap_paf_1, all_peaks_1, img_shape, out_form=1, idx_channel=2)

        heatmap_kpt_2 = ht2[0, :self.num_keypoints_2, :, :]
        all_peaks_2, all_scroes_2 = get_all_peaks(heatmap_kpt_2, sigma=self.sigma_2, idx_channel=0, upscale=8)


        heatmap_kpt_3 = ht3[0, :self.num_keypoints_3, :, :]
        all_peaks_3, all_scroes_3 = get_all_peaks(heatmap_kpt_3, sigma=self.sigma_3, idx_channel=0, upscale=4)

        heatmap_paf_3 = ht3[0, self.num_keypoints_3:, :, :]
        heatmap_paf_3 = np.transpose(heatmap_paf_3, (1, 2, 0))
        heatmap_paf_3 = cv2.resize(heatmap_paf_3, (img_shape, img_shape), interpolation=cv2.INTER_CUBIC)
        all_skeletons_3 = get_all_skeletons(heatmap_paf_3, all_peaks_3, img_shape, out_form=3, idx_channel=2)

        return all_peaks_1, all_scroes_1, None,  all_peaks_2, all_scroes_2, None, all_peaks_3, all_scroes_3, all_skeletons_3

        # keypoints, scores = get_heatmap_maximum(heatmaps)
        #
        # # Unsqueeze the instance dimension for single-instance results
        # keypoints, scores = keypoints[None], scores[None]
        #
        # if self.unbiased:
        #     # Alleviate biased coordinate
        #     keypoints = refine_keypoints_dark(
        #         keypoints, heatmaps, blur_kernel_size=self.blur_kernel_size)
        #
        # else:
        #     keypoints = refine_keypoints(keypoints, heatmaps)
        #
        # # Restore the keypoint scale
        # keypoints = keypoints * self.scale_factor
        #
        # return keypoints, scores
