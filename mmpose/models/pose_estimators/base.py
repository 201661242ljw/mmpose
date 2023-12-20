# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch
from mmengine.dist import get_world_size
from mmengine.logging import print_log
from mmengine.model import BaseModel
from torch import Tensor
import math

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.models.utils import check_and_update_config
from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, ForwardResults, OptConfigType,
                                 Optional, OptMultiConfig, OptSampleList,
                                 SampleList)

import colorsys


class BasePoseEstimator(BaseModel, metaclass=ABCMeta):
    """Base class for pose estimators.

    Args:
        data_preprocessor (dict | ConfigDict, optional): The pre-processing
            config of :class:`BaseDataPreprocessor`. Defaults to ``None``
        init_cfg (dict | ConfigDict): The model initialization config.
            Defaults to ``None``
        use_syncbn (bool): whether to use SyncBatchNorm. Defaults to False.
        metainfo (dict): Meta information for dataset, such as keypoints
            definition and properties. If set, the metainfo of the input data
            batch will be overridden. For more details, please refer to
            https://mmpose.readthedocs.io/en/latest/user_guides/
            prepare_datasets.html#create-a-custom-dataset-info-
            config-file-for-the-dataset. Defaults to ``None``
    """
    _version = 2

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 use_syncbn: bool = False,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.metainfo = self._load_metainfo(metainfo)
        self.train_cfg = train_cfg if train_cfg else {}
        self.test_cfg = test_cfg if test_cfg else {}

        self.backbone = MODELS.build(backbone)

        # the PR #2108 and #2126 modified the interface of neck and head.
        # The following function automatically detects outdated
        # configurations and updates them accordingly, while also providing
        # clear and concise information on the changes made.
        neck, head = check_and_update_config(neck, head)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if head is not None:
            self.head = MODELS.build(head)
            self.head.test_cfg = self.test_cfg.copy()

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

        # TODO： Waiting for mmengine support
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current')

    @property
    def with_neck(self) -> bool:
        """bool: whether the pose estimator has a neck."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        """bool: whether the pose estimator has a head."""
        return hasattr(self, 'head') and self.head is not None

    @staticmethod
    def _load_metainfo(metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (dict): Raw data of pose meta information.

        Returns:
            dict: Parsed meta information.
        """

        if metainfo is None:
            return None

        if not isinstance(metainfo, dict):
            raise TypeError(
                f'metainfo should be a dict, but got {type(metainfo)}')

        metainfo = parse_pose_metainfo(metainfo)
        return metainfo

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: 'tensor', 'predict' and 'loss':

        - 'tensor': Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - 'predict': Forward and return the predictions, which are fully
        processed to a list of :obj:`PoseDataSample`.
        - 'loss': Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general
            data_samples (list[:obj:`PoseDataSample`], optional): The
                annotation of every sample. Defaults to ``None``
            mode (str): Set the forward mode and return value type. Defaults
                to ``'tensor'``

        Returns:
            The return type depends on ``mode``.

            - If ``mode='tensor'``, return a tensor or a tuple of tensors
            - If ``mode='predict'``, return a list of :obj:``PoseDataSample``
                that contains the pose predictions
            - If ``mode='loss'``, return a dict of tensor(s) which is the loss
                function value
        """
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            # use customed metainfo to override the default metainfo
            if self.metainfo is not None:
                for data_sample in data_samples:
                    data_sample.set_metainfo(self.metainfo)
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs)
        elif mode == 'LJW_show_heatmap_and_pafs':
            if not True:
            # if True:
                if self.metainfo is not None:
                    for data_sample in data_samples:
                        data_sample.set_metainfo(self.metainfo)
                return self.predict(inputs, data_samples)


            outputs = self._forward(inputs)

            bgr_colors = []
            hue_steps = 22  # 均分的色调数量
            saturation = 1.0  # 最大饱和度
            value = 1.0  # 最大明度
            for i in range(hue_steps):
                hue = i / hue_steps  # 计算色调
                rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)  # 转换为RGB颜色
                bgr_color = tuple([round(x * 255) for x in reversed(rgb_color)])  # 转换为BGR颜色
                bgr_colors.append(bgr_color)
            sk_colores = bgr_colors

            for (idx, data_sample) in enumerate(data_samples):
                img_name = os.path.basename(data_sample.img_path)
                # if not "04_2_066_head_have_2_0" in img_name:
                #     continue
                save_dir = r"E:\LJW\Git\mmpose\tools\0_LJW_tools\predict_show_1024_temp_7/{}".format(
                    img_name.split(".")[0])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                tensor = inputs[idx, :, :, :].detach().cpu()
                img = tensor.numpy()
                img = img.transpose(1, 2, 0) * [58.395, 57.12, 57.375] + [123.675, 116.28, 103.53]
                img = img.astype('uint8')
                img = img[:, :, ::-1]  # BGR to RGB
                img = np.ascontiguousarray(img)  # channel-first to row-first
                org_img_path = os.path.join(save_dir, img_name)
                # cv2.imwrite(org_img_path, img)

                output = outputs[idx, :, :, :].detach().cpu().numpy()[-49:]
                for i in range(output.shape[0]):
                    ht = output[i]
                    r = np.clip(ht, 0, 1) * 255
                    g = ht * 0
                    b = np.clip(-ht, 0, 1) * 255
                    ht = np.array([b, g, r], dtype=np.uint8).transpose(1, 2, 0)
                    if i < 5:
                        save_path = os.path.join(save_dir, "ht_{}.jpg".format(i + 1))
                    else:
                        save_path = os.path.join(save_dir, "paf_{}_{}.jpg".format((i - 5) // 2 + 1, 'xy'[(i - 5) % 2]))

                    ht = cv2.resize(ht, (img.shape[1], img.shape[0]))

                    cv2.imwrite(save_path, (ht * 0.75 + img * 0.25).astype(np.uint8))

                heatmap_avg = output[:5]
                heatmap_avg = np.transpose(heatmap_avg, (1, 2, 0))
                paf_avg = output[5:]
                paf_avg = np.transpose(paf_avg, (1, 2, 0))
                heatmap_avg = cv2.resize(heatmap_avg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
                paf_avg = cv2.resize(paf_avg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

                all_peaks = []
                peak_counter = 0
                thre1 = 0.1
                thre2 = 0.05
                for part in range(heatmap_avg.shape[2]):
                    map_ori = heatmap_avg[:, :, part]
                    one_heatmap = gaussian_filter(map_ori, sigma=2)

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
                    peak_counter += len(peaks)

                pt_colors = [[0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 0], [255, 0, 255]]
                # sk_colores = [[0, 255, 128], [0, 128, 255], [255, 0, 128], [128, 0, 255], [255, 8128, 0]]

                pt_img = copy.deepcopy(img)
                pt_img = pt_img // 4
                for (all_peak, pt_color) in zip(all_peaks, pt_colors):
                    for pt in all_peak:
                        x = pt[0]
                        y = pt[1]
                        pt_img = cv2.circle(pt_img, center=(x, y), color=pt_color, radius=5, thickness=-1)

                save_path = os.path.join(save_dir, "0_pts.jpg")

                cv2.imwrite(save_path, pt_img)

                # find connection in the specified sequence, center 29 is in the position 15
                # limbSeq = [[1, 0],
                #            [1, 1],
                #            [1, 2],
                #            [1, 3],
                #            [4, 0]]
                limbSeq = [
                    [1, 0],
                    [1, 0],
                    [0, 0],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 2],
                    [2, 2],
                    [1, 3],
                    [1, 3],
                    [3, 3],
                    [1, 3],
                    [4, 0],
                    [4, 0],
                    [0, 0],
                    [4, 4],
                    [4, 4],
                    [4, 4],
                ]
                # the middle joints heatmap correpondence
                mapIdx = [
                    [0, 1],
                    [2, 3],
                    [4, 5],
                    [6, 7],
                    [8, 9],
                    [10, 11],
                    [12, 13],
                    [14, 15],
                    [16, 17],
                    [18, 19],
                    [20, 21],
                    [22, 23],
                    [24, 25],
                    [26, 27],
                    [28, 29],
                    [30, 31],
                    [32, 33],
                    [34, 35],
                    [36, 37],
                    [38, 39],
                    [40, 41],
                    [42, 43]
                ]

                connection_all = []
                special_connection = []
                # special_k = []
                mid_num = 10

                for k in range(len(mapIdx)):
                    score_mid = paf_avg[:, :, [x for x in mapIdx[k]]]
                    candA = all_peaks[limbSeq[k][0]]
                    candB = all_peaks[limbSeq[k][1]]
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

                                startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                                    np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                                vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                                  for I in range(len(startend))])
                                vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                                  for I in range(len(startend))])

                                score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                                score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                                    0.5 * img.shape[0] / norm - 1, 0)
                                criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                                criterion2 = score_with_dist_prior > 0
                                if criterion1 and criterion2:
                                    # pt1_type,pt1_id,pt1_x,pt1_y
                                    # pt2_type,pt2_id,pt2_x,pt2_y
                                    # length, vec_x, vec_y, skeleton_type,skeleton_score, skeelton_and_pts_score
                                    skeleton = [limbSeq[k][0] + 1, candA[i][2], candA[i][0], candA[i][1],
                                                limbSeq[k][1] + 1, candB[j][2], candB[j][0], candB[j][1],
                                                norm, vec[0], vec[1], k + 1, score_with_dist_prior,
                                                score_with_dist_prior + candA[i][2] + candB[j][2]]
                                    # if k == 1:
                                    #     need_record = True
                                    #     if len(special_connection) != 0:
                                    #         for (idx, skeleton_temp) in enumerate(special_connection):
                                    #             if skeleton_temp[1] == candA[i][2]:
                                    #                 if vec[0] * skeleton_temp[9] + vec[1] * skeleton_temp[10] > 0.9:
                                    #                     need_record = False
                                    #                     if norm < skeleton_temp[8]:
                                    #                         special_connection[idx] = skeleton
                                    #     if need_record:
                                    #         special_connection.append(skeleton)
                                    # else:
                                    #     connection_all.append(skeleton)
                                    if skeleton[-1] > max_temp:
                                        connection_candidate = skeleton
                                        max_temp = skeleton[-1]
                            if connection_candidate is not None:
                                connection_all.append(connection_candidate)

                connection_all = connection_all + special_connection
                pt_drawn = []
                img = img // 4
                for [pt1_type, pt1_id, pt1_x, pt1_y, pt2_type, pt2_id, pt2_x, pt2_y, length, vec_x, vec_y,
                     skeleton_type, skeleton_score, skeelton_and_pts_score] in connection_all:
                    if not pt1_id in pt_drawn:
                        img = cv2.circle(img, center=(pt1_x, pt1_y), radius=5, color=pt_colors[pt1_type - 1],
                                         thickness=-1)
                        pt_drawn.append(pt1_id)
                    if not pt2_id in pt_drawn:
                        img = cv2.circle(img, center=(pt2_x, pt2_y), radius=5, color=pt_colors[pt2_type - 1],
                                         thickness=-1)
                        pt_drawn.append(pt2_id)
                    img = cv2.line(img, (pt1_x, pt1_y), (pt2_x, pt2_y), thickness=2,
                                   color=sk_colores[skeleton_type - 1])
                cv2.imwrite(os.path.join(save_dir, "0_show.jpg"), img)
                # exit()
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode.')

    @abstractmethod
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

    @abstractmethod
    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None
                 ) -> Union[Tensor, Tuple[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            Union[Tensor | Tuple[Tensor]]: forward output of the network.
        """

        x = self.extract_feat(inputs)
        if self.with_head:
            x = self.head.forward(x)

        return x

    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)

        return x

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to.

        1) convert old-version state dict of
        :class:`TopdownHeatmapSimpleHead` (before MMPose v1.0.0) to a
        compatible format of :class:`HeatmapHead`.

        2) remove the weights in data_preprocessor to avoid warning
        `unexpected key in source state_dict: ...`. These weights are
        initialized with given arguments and remain same during training
        and inference.

        The hook will be automatically registered during initialization.
        """

        keys = list(state_dict.keys())

        # remove the keys in data_preprocessor to avoid warning
        for k in keys:
            if k in ('data_preprocessor.mean', 'data_preprocessor.std'):
                del state_dict[k]

        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        for k in keys:
            if 'keypoint_head' in k:
                v = state_dict.pop(k)
                k = k.replace('keypoint_head', 'head')
                state_dict[k] = v
