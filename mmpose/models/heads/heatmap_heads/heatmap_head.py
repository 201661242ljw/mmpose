# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
from typing import Optional, Sequence, Tuple, Union

import torch
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.evaluation.functional import pose_pck_accuracy, ljw_tower_pose_pack_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from mmpose.utils import get_all_peaks, get_all_skeletons
import cv2


@MODELS.register_module()
class HeatmapHead(BaseHead):
    """Top-down heatmap head introduced in `Simple Baselines`_ by Xiao et al
    (2018). The head is composed of a few deconvolutional layers followed by a
    convolutional layer to generate heatmaps from low-resolution feature maps.

    Args:
        in_channels (int | Sequence[int]): Number of channels in the input
            feature map
        out_channels (int): Number of channels in the output heatmap
        deconv_out_channels (Sequence[int], optional): The output channel
            number of each deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each deconv layer. Each element should be either an integer for
            both height and width dimensions, or a tuple of two integers for
            the height and the width dimension respectively.Defaults to
            ``(4, 4, 4)``
        conv_out_channels (Sequence[int], optional): The output channel number
            of each intermediate conv layer. ``None`` means no intermediate
            conv layer between deconv layers and the final conv layer.
            Defaults to ``None``
        conv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each intermediate conv layer. Defaults to ``None``
        final_layer (dict): Arguments of the final Conv2d layer.
            Defaults to ``dict(kernel_size=1)``
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KeypointMSELoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`Simple Baselines`: https://arxiv.org/abs/1804.06208
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 deconv_out_channels: OptIntSeq = (256, 256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 final_layer: dict = dict(kernel_size=1),
                 loss: ConfigType = dict(
                     type='KeypointMSELoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                    deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {deconv_out_channels} and '
                    f'{deconv_kernel_sizes}')

            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(
                    conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {conv_out_channels} and '
                    f'{conv_kernel_sizes}')

            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        if final_layer is not None:
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)
            cfg.update(final_layer)
            self.final_layer = build_conv_layer(cfg)
        else:
            self.final_layer = nn.Identity()

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
            layers.append(build_conv_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')
            cfg = dict(
                type='deconv',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            layers.append(build_upsample_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        x = feats[-1]

        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        x = self.final_layer(x)

        return x

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
            ``test_cfg['output_heatmap']==True``, return both pose and heatmap
            prediction; otherwise only return the pose prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _batch_heatmaps = self.forward(_feats)
            _batch_heatmaps_flip = flip_heatmaps(
                self.forward(_feats_flip),
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
        else:
            batch_heatmaps = self.forward(feats)

        preds = self.decode(batch_heatmaps)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]
            return preds, pred_fields
        else:
            return preds

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            train_cfg (dict): The runtime config for training process.
                Defaults to {}

        Returns:
            dict: A dictionary of losses.
        """
        pred_fields = self.forward(feats)
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)

        losses.update(loss_kpt=loss)

        # calculate accuracy
        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0)

            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

        return losses

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`TopdownHeatmapSimpleHead` (before MMPose v1.0.0) to a
        compatible format of :class:`HeatmapHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for _k in keys:
            if not _k.startswith(prefix):
                continue
            v = state_dict.pop(_k)
            k = _k[len(prefix):]
            # In old version, "final_layer" includes both intermediate
            # conv layers (new "conv_layers") and final conv layers (new
            # "final_layer").
            #
            # If there is no intermediate conv layer, old "final_layer" will
            # have keys like "final_layer.xxx", which should be still
            # named "final_layer.xxx";
            #
            # If there are intermediate conv layers, old "final_layer"  will
            # have keys like "final_layer.n.xxx", where the weights of the last
            # one should be renamed "final_layer.xxx", and others should be
            # renamed "conv_layers.n.xxx"
            k_parts = k.split('.')
            if k_parts[0] == 'final_layer':
                if len(k_parts) == 3:
                    assert isinstance(self.conv_layers, nn.Sequential)
                    idx = int(k_parts[1])
                    if idx < len(self.conv_layers):
                        # final_layer.n.xxx -> conv_layers.n.xxx
                        k_new = 'conv_layers.' + '.'.join(k_parts[1:])
                    else:
                        # final_layer.n.xxx -> final_layer.xxx
                        k_new = 'final_layer.' + k_parts[2]
                else:
                    # final_layer.xxx remains final_layer.xxx
                    k_new = k
            else:
                k_new = k

            state_dict[prefix + k_new] = v


@MODELS.register_module()
class LJW_HeatmapHead(BaseHead):
    """Top-down heatmap head introduced in `Simple Baselines`_ by Xiao et al
    (2018). The head is composed of a few deconvolutional layers followed by a
    convolutional layer to generate heatmaps from low-resolution feature maps.

    Args:
        in_channels (int | Sequence[int]): Number of channels in the input
            feature map
        out_channels (int): Number of channels in the output heatmap
        deconv_out_channels (Sequence[int], optional): The output channel
            number of each deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each deconv layer. Each element should be either an integer for
            both height and width dimensions, or a tuple of two integers for
            the height and the width dimension respectively.Defaults to
            ``(4, 4, 4)``
        conv_out_channels (Sequence[int], optional): The output channel number
            of each intermediate conv layer. ``None`` means no intermediate
            conv layer between deconv layers and the final conv layer.
            Defaults to ``None``
        conv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each intermediate conv layer. Defaults to ``None``
        final_layer (dict): Arguments of the final Conv2d layer.
            Defaults to ``dict(kernel_size=1)``
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KeypointMSELoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`Simple Baselines`: https://arxiv.org/abs/1804.06208
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 use_medium_satge,
                 heatmap_scale,
                 target_form,
                 output_form_1,
                 output_form_2,
                 output_form_3,
                 channel_labels,
                 deconv_out_channels: OptIntSeq = (256, 256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 final_layer: dict = dict(kernel_size=1),
                 loss: ConfigType = dict(
                     type='KeypointMSELoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)
        self.heatmap_scale = heatmap_scale
        self.output_form_1 = output_form_1
        self.output_form_2 = output_form_2
        self.output_form_3 = output_form_3
        self.use_medium_satge = use_medium_satge
        self.target_form = target_form
        self.channel_labels = channel_labels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_module = MODELS.build(loss)

        self.deconv_layer_12 = self._make_deconv_layers(384, [192], [4])
        self.deconv_layer_23 = self._make_deconv_layers(399, [96], [4])
        self.deconv_layer_34 = self._make_deconv_layers(207, [48], [4])

        self.conv_layer_1 = self._make_conv_layers(384, [256, 256], [3, 3])
        self.conv_layer_2 = self._make_conv_layers(192, [256, 256], [3, 3])
        self.conv_layer_3 = self._make_conv_layers(96, [256, 256], [3, 3])

        self.ht_layer_1 = nn.Conv2d(256, 15, 1, 1)
        self.ht_layer_2 = nn.Conv2d(256, 15, 1, 1)
        self.ht_layer_3 = nn.Conv2d(256, 92, 1, 1)

        # self.use_medium_satge = use_medium_satge
        # self.target_form = target_form
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                    deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {deconv_out_channels} and '
                    f'{deconv_kernel_sizes}')

            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(
                    conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {conv_out_channels} and '
                    f'{conv_kernel_sizes}')

            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        if final_layer is not None:
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)
            cfg.update(final_layer)
            self.final_layer = build_conv_layer(cfg)
        else:
            self.final_layer = nn.Identity()

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
            layers.append(build_conv_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')
            cfg = dict(
                type='deconv',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            layers.append(build_upsample_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg

    def forward(self, feats: list) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        (x4, x3, x2, x1) = feats

        x12 = self.deconv_layer_12(x1)
        x212 = torch.cat([x2, x12], 1)
        y1 = self.ht_layer_1(self.conv_layer_1(x212))
        x212y1 = torch.cat([x212, y1], 1)

        x23 = self.deconv_layer_23(x212y1)
        x323 = torch.cat([x3, x23], 1)
        y2 = self.ht_layer_2(self.conv_layer_2(x323))
        x323y2 = torch.cat([x323, y2], 1)

        x34 = self.deconv_layer_34(x323y2)
        x434 = torch.cat([x4, x34], 1)
        y3 = self.ht_layer_3(self.conv_layer_3(x434))

        return (y1, y2, y3)

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
            ``test_cfg['output_heatmap']==True``, return both pose and heatmap
            prediction; otherwise only return the pose prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
        """
        #
        # if test_cfg.get('flip_test', False):
        #     # TTA: flip test -> feats = [orig, flipped]
        #     assert isinstance(feats, list) and len(feats) == 2
        #     flip_indices = batch_data_samples[0].metainfo['flip_indices']
        #     _feats, _feats_flip = feats
        #     _batch_heatmaps = self.forward(_feats)
        #     _batch_heatmaps_flip = flip_heatmaps(
        #         self.forward(_feats_flip),
        #         flip_mode=test_cfg.get('flip_mode', 'heatmap'),
        #         flip_indices=flip_indices,
        #         shift_heatmap=test_cfg.get('shift_heatmap', False))
        #     batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
        # else:
        batch_heatmaps = self.forward(feats)

        preds = self.decode(batch_heatmaps)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]
            return preds, pred_fields
        else:
            return preds

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            train_cfg (dict): The runtime config for training process.
                Defaults to {}

        Returns:
            dict: A dictionary of losses.
        """
        pred_fields = self.forward(feats)

        gt_heatmaps_1 = torch.stack([d.heatmaps_1 for d in batch_data_samples]).to(torch.device(feats[0].device))
        gt_heatmaps_2 = torch.stack([d.heatmaps_2 for d in batch_data_samples]).to(torch.device(feats[0].device))
        gt_heatmaps_3 = torch.stack([d.heatmaps_3 for d in batch_data_samples]).to(torch.device(feats[0].device))

        keypoint_weights_1 = torch.cat([d.keypoint_weights_1 for d in batch_data_samples])
        keypoint_weights_2 = torch.cat([d.keypoint_weights_2 for d in batch_data_samples])
        keypoint_weights_3 = torch.cat([d.keypoint_weights_3 for d in batch_data_samples])

        # gt_heatmaps = torch.stack(
        #     [d.gt_fields.heatmaps for d in batch_data_samples])
        # keypoint_weights = torch.cat([
        #     d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        # ])

        # calculate losses
        losses = dict()
        # loss = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)

        kt1 = self.channel_labels[0][0] * self.channel_labels[0][2] * self.channel_labels[2][2]
        sk1 = self.channel_labels[0][1] * self.channel_labels[0][2] * self.channel_labels[2][2] * 2
        kt2 = self.channel_labels[1][0] * self.channel_labels[1][2] * self.channel_labels[2][2]
        sk2 = self.channel_labels[1][1] * self.channel_labels[1][2] * self.channel_labels[2][2] * 2
        kt3 = self.channel_labels[2][0] * self.channel_labels[2][2] * self.channel_labels[2][2]
        sk3 = self.channel_labels[2][1] * self.channel_labels[2][2] * self.channel_labels[2][2] * 2

        # ----------------------------------------------------------------------------------------------------
        # 1
        # ----------------------------------------------------------------------------------------------------
        if self.output_form_1:
            losses.update(
                loss_p1=self.loss_module(
                    pred_fields[0][:, :kt1, :, :],
                    gt_heatmaps_1[:, :kt1, :, :],
                    keypoint_weights_1[:, :kt1]
                ) * 0.2
            )
            losses.update(
                loss_s1=self.loss_module(
                    pred_fields[0][:, kt1:, :, :],
                    gt_heatmaps_1[:, kt1:, :, :],
                    keypoint_weights_1[:, kt1:]
                ) * 0.2
            )
        # ----------------------------------------------------------------------------------------------------
        # 2
        # ----------------------------------------------------------------------------------------------------
        if self.output_form_2:
            losses.update(
                loss_p2=self.loss_module(
                    pred_fields[1][:, :kt2, :, :],
                    gt_heatmaps_2[:, :kt2, :, :],
                    keypoint_weights_2[:, :kt2]
                ) * 0.4
            )
            losses.update(
                loss_s2=self.loss_module(
                    pred_fields[1][:, kt2:, :, :],
                    gt_heatmaps_2[:, kt2:, :, :],
                    keypoint_weights_2[:, kt2:]
                ) * 0.4
            )
        # ----------------------------------------------------------------------------------------------------
        # 3
        # ----------------------------------------------------------------------------------------------------
        if self.output_form_3:
            losses.update(
                loss_p3=self.loss_module(
                    pred_fields[2][:, :kt3, :, :],
                    gt_heatmaps_3[:, :kt3, :, :],
                    keypoint_weights_3[:, :kt3]
                )
            )
            losses.update(
                loss_s3=self.loss_module(
                    pred_fields[2][:, kt3:, :, :],
                    gt_heatmaps_3[:, kt3:, :, :],
                    keypoint_weights_3[:, kt3:]
                )
            )
        # ----------------------------------------------------------------------------------------------------

        # calculate accuracy
        if train_cfg.get('compute_acc', True):

            for idx, x in enumerate(batch_data_samples):

                # -------------------------------------------------------------------------------------------------------
                # 1
                # -------------------------------------------------------------------------------------------------------
                gt_kt_1 = [[] for _ in range(self.channel_labels[0][0])]
                for kt_idx, [x_, y_, type] in enumerate(x.raw_ann_info['true_points_1']):
                    gt_kt_1[type - 1].append([x_ / self.heatmap_scale[0], y_ / self.heatmap_scale[0], kt_idx])

                output = pred_fields[0][idx, :, :, :].detach().cpu().numpy()
                heatmap_avg = output[:kt1]
                pred_kt_1, _ = get_all_peaks(heatmap_avg, sigma=self.channel_labels[0][3], idx_channel=0)
                tps_1, fps_1, fns_1 = ljw_tower_pose_pack_accuracy(
                    output=pred_kt_1,
                    target=gt_kt_1,
                    sigma=self.channel_labels[0][3] * 2
                )
                # paf_avg = output[kt1:]
                # paf_avg = np.transpose(paf_avg, (1, 2, 0))
                # paf_avg = cv2.resize(paf_avg, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
                # for part in range(paf_avg.shape[2]):
                #     ht = paf_avg[:,:,part]
                #     r = np.clip(ht, 0, 1) * 255
                #     g = ht * 0
                #     b = np.clip(-ht, 0, 1) * 255
                #     ht = np.array([b, g, r], dtype=np.uint8).transpose(1, 2, 0)
                #     save_path = r"E:\LJW\Git\mmpose\tools\0_LJW_tools\val\1_{}_{}.jpg".format(part // 2, "xy"[part % 2])
                #     cv2.imwrite(save_path, ht)
                # gt_sk_1 = [[] for _ in range(self.channel_labels[0][1])]
                # for sk_idx, [p1_x, p1_y, p1_t, p2_x, p2_y, p2_t, type] in enumerate(x.raw_ann_info['true_skeletons_1']):
                #     gt_sk_1[type - 1].append(
                #         [p1_x, p1_y, p1_t,
                #          p2_x, p2_y, p2_t,
                #          type - 1
                #          ]
                #     )
                # all_skeletons_1 = get_all_skeletons(paf_avg, pred_kt_1, img_w, out_form=1, idx_channel=2)
                # -------------------------------------------------------------------------------------------------------
                # 2
                # -------------------------------------------------------------------------------------------------------
                gt_kt_2 = [[] for _ in range(self.channel_labels[1][0])]
                for kt_idx, [x_, y_, type] in enumerate(x.raw_ann_info['true_points_2']):
                    gt_kt_2[type - 1].append([x_ / self.heatmap_scale[1], y_ / self.heatmap_scale[1], kt_idx])

                output = pred_fields[1][idx, :, :, :].detach().cpu().numpy()
                heatmap_avg = output[:kt2]
                # heatmap_avg = np.transpose(heatmap_avg, (1, 2, 0))
                # heatmap_avg = cv2.resize(heatmap_avg, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

                pred_kt_2, _ = get_all_peaks(heatmap_avg, sigma=self.channel_labels[1][3], idx_channel=0)
                tps_2, fps_2, fns_2 = ljw_tower_pose_pack_accuracy(
                    output=pred_kt_2,
                    target=gt_kt_2,
                    sigma=self.channel_labels[1][3] * 2
                )
                #
                # gt_sk_2 = [[] for _ in range(self.channel_labels[1][1])]
                # for sk_idx, [p1_x, p1_y, p1_t, p2_x, p2_y, p2_t, type] in enumerate(x.raw_ann_info['true_skeletons_2']):
                #     gt_sk_2[type - 1].append(
                #         [p1_x / self.heatmap_scale[1], p1_y / self.heatmap_scale[1], p1_t,
                #          p2_x / self.heatmap_scale[1], p2_y / self.heatmap_scale[1], p2_t,
                #          type - 1
                #          ]
                #     )
                # img_shape = pred_fields[1][idx, kt1:, :, :].shape[1]
                # all_skeletons_2 = get_all_skeletons(to_numpy(pred_fields[1][idx, kt2:, :, :]), pred_kt_2, img_shape,
                #                                     out_form=2)

                # -------------------------------------------------------------------------------------------------------
                # 3
                # -------------------------------------------------------------------------------------------------------
                gt_kt_3 = [[] for _ in range(self.channel_labels[2][0])]
                for kt_idx, [x_, y_, type] in enumerate(x.raw_ann_info['true_points_3']):
                    gt_kt_3[type - 1].append([x_ / self.heatmap_scale[2], y_ / self.heatmap_scale[2], kt_idx])

                output = pred_fields[2][idx, :, :, :].detach().cpu().numpy()
                heatmap_avg = output[:kt3]
                # heatmap_avg = np.transpose(heatmap_avg, (1, 2, 0))
                # heatmap_avg = cv2.resize(heatmap_avg, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
                #
                pred_kt_3, _ = get_all_peaks(heatmap_avg, sigma=self.channel_labels[2][3], idx_channel=0)
                tps_3, fps_3, fns_3 = ljw_tower_pose_pack_accuracy(
                    output=pred_kt_3,
                    target=gt_kt_3,
                    sigma=self.channel_labels[2][3] * 2
                )
                # gt_sk_3 = [[] for _ in range(self.channel_labels[2][1])]
                # for sk_idx, [p1_x, p1_y, p1_t, p2_x, p2_y, p2_t, type] in enumerate(x.raw_ann_info['true_skeletons_3']):
                #     gt_sk_3[type - 1].append(
                #         [p1_x / self.heatmap_scale[2], p1_y / self.heatmap_scale[2], p1_t,
                #          p2_x / self.heatmap_scale[2], p2_y / self.heatmap_scale[2], p2_t,
                #          type - 1
                #          ]
                #     )
                # img_shape = pred_fields[2][idx, kt1:, :, :].shape[1]
                # all_skeletons_3 = get_all_skeletons(to_numpy(pred_fields[2][idx, kt3:, :, :]), pred_kt_3, img_shape)

                # -------------------------------------------------------------------------------------------------------

                log_path = r"E:\LJW\Git\mmpose\tools\0_LJW_tools\_pose_acc_log.json"
                if not os.path.exists(log_path):
                    log_data = {
                        "i": 1,
                        "tps_1": tps_1,
                        "fps_1": fps_1,
                        "fns_1": fns_1,
                        "tps_2": tps_2,
                        "fps_2": fps_2,
                        "fns_2": fns_2,
                        "tps_3": tps_3,
                        "fps_3": fps_3,
                        "fns_3": fns_3,
                    }
                else:
                    log_data = json.load(open(log_path, "r", encoding="utf-8"), strict=False)
                    log_data['i'] += 1
                    log_data["tps_1"] += tps_1
                    log_data["fps_1"] += fps_1
                    log_data["fns_1"] += fns_1
                    log_data["tps_2"] += tps_2
                    log_data["fps_2"] += fps_2
                    log_data["fns_2"] += fns_2
                    log_data["tps_3"] += tps_3
                    log_data["fps_3"] += fps_3
                    log_data["fns_3"] += fns_3

                precision_1 = log_data["tps_1"] / max(1, (log_data["tps_1"] + log_data["fps_1"]))
                recall_1 = log_data["tps_1"] / max(1, (log_data["tps_1"] + log_data["fns_1"]))
                f1_score_1 = 2 * (precision_1 * recall_1) / max(1e-4, (precision_1 + recall_1))

                precision_2 = log_data["tps_2"] / max(1, (log_data["tps_2"] + log_data["fps_2"]))
                recall_2 = log_data["tps_2"] / max(1, (log_data["tps_2"] + log_data["fns_2"]))
                f1_score_2 = 2 * (precision_2 * recall_2) / max(1e-4, (precision_2 + recall_2))

                precision_3 = log_data["tps_3"] / max(1, (log_data["tps_3"] + log_data["fps_3"]))
                recall_3 = log_data["tps_3"] / max(1, (log_data["tps_3"] + log_data["fns_3"]))
                f1_score_3 = 2 * (precision_3 * recall_3) / max(1e-4, (precision_3 + recall_3))

                losses.update(pp1=torch.tensor(precision_1, device=feats[0].device))
                losses.update(pr1=torch.tensor(recall_1, device=feats[0].device))
                losses.update(pf1=torch.tensor(f1_score_1, device=feats[0].device))
                losses.update(pp2=torch.tensor(precision_2, device=feats[0].device))
                losses.update(pr2=torch.tensor(recall_2, device=feats[0].device))
                losses.update(pf2=torch.tensor(f1_score_2, device=feats[0].device))
                losses.update(pp3=torch.tensor(precision_3, device=feats[0].device))
                losses.update(pr3=torch.tensor(recall_3, device=feats[0].device))
                losses.update(pf3=torch.tensor(f1_score_3, device=feats[0].device))

                acc_pose = torch.tensor(f1_score_1 * 0.15 + f1_score_2 * 0.2 + f1_score_3 * 0.65,
                                        device=feats[0].device)
                losses.update(acc_pose=acc_pose)

        return losses

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`TopdownHeatmapSimpleHead` (before MMPose v1.0.0) to a
        compatible format of :class:`HeatmapHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for _k in keys:
            if not _k.startswith(prefix):
                continue
            v = state_dict.pop(_k)
            k = _k[len(prefix):]
            # In old version, "final_layer" includes both intermediate
            # conv layers (new "conv_layers") and final conv layers (new
            # "final_layer").
            #
            # If there is no intermediate conv layer, old "final_layer" will
            # have keys like "final_layer.xxx", which should be still
            # named "final_layer.xxx";
            #
            # If there are intermediate conv layers, old "final_layer"  will
            # have keys like "final_layer.n.xxx", where the weights of the last
            # one should be renamed "final_layer.xxx", and others should be
            # renamed "conv_layers.n.xxx"
            k_parts = k.split('.')
            if k_parts[0] == 'final_layer':
                if len(k_parts) == 3:
                    assert isinstance(self.conv_layers, nn.Sequential)
                    idx = int(k_parts[1])
                    if idx < len(self.conv_layers):
                        # final_layer.n.xxx -> conv_layers.n.xxx
                        k_new = 'conv_layers.' + '.'.join(k_parts[1:])
                    else:
                        # final_layer.n.xxx -> final_layer.xxx
                        k_new = 'final_layer.' + k_parts[2]
                else:
                    # final_layer.xxx remains final_layer.xxx
                    k_new = k
            else:
                k_new = k

            state_dict[prefix + k_new] = v
