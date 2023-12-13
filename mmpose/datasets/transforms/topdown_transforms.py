# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine import is_seq_of

from mmpose.registry import TRANSFORMS
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix


@TRANSFORMS.register_module()
class TopdownAffine(BaseTransform):
    """Get the bbox image as the model input by affine transform.

    Required Keys:

        - img
        - bbox_center
        - bbox_scale
        - bbox_rotation (optional)
        - keypoints (optional)

    Modified Keys:

        - img
        - bbox_scale

    Added Keys:

        - input_size
        - transformed_keypoints

    Args:
        input_size (Tuple[int, int]): The input image size of the model in
            [w, h]. The bbox region will be cropped and resize to `input_size`
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 use_udp: bool = False) -> None:
        super().__init__()

        assert is_seq_of(input_size, int) and len(input_size) == 2, (
            f'Invalid input_size {input_size}')

        self.input_size = input_size
        self.use_udp = use_udp

    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
        """Reshape the bbox to a fixed aspect ratio.

        Args:
            bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.darray: The reshaped bbox scales in (n, 2)
        """

        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                              np.hstack([w, w / aspect_ratio]),
                              np.hstack([h * aspect_ratio, h]))
        return bbox_scale

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`TopdownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        w, h = self.input_size
        warp_size = (int(w), int(h))

        # reshape bbox to fixed aspect ratio
        results['bbox_scale'] = self._fix_aspect_ratio(
            results['bbox_scale'], aspect_ratio=w / h)

        # TODO: support multi-instance
        assert results['bbox_center'].shape[0] == 1, (
            'Top-down heatmap only supports single instance. Got invalid '
            f'shape of bbox_center {results["bbox_center"].shape}.')

        center = results['bbox_center'][0]
        scale = results['bbox_scale'][0]
        if 'bbox_rotation' in results:
            rot = results['bbox_rotation'][0]
        else:
            rot = 0.

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                center, scale, rot, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

        if isinstance(results['img'], list):
            results['img'] = [
                cv2.warpAffine(
                    img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                for img in results['img']
            ]
        else:
            # 图像变换
            results['img'] = cv2.warpAffine(results['img'], warp_mat, warp_size, flags=cv2.INTER_LINEAR)

        if results.get('keypoints', None) is not None:
            # 关键点变换
            transformed_keypoints = results['keypoints'].copy()
            # Only transform (x, y) coordinates
            transformed_keypoints[..., :2] = cv2.transform(results['keypoints'][..., :2], warp_mat)

            # 真实标注关键点变化
            results['transformed_keypoints'] = transformed_keypoints

        results['input_size'] = (w, h)
        results['input_center'] = center
        results['input_scale'] = scale  # scale = max(bbox_w, bbox_h) * padding


        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'use_udp={self.use_udp})'
        return repr_str


@TRANSFORMS.register_module()
class LJW_TopdownAffine(BaseTransform):
    """Get the bbox image as the model input by affine transform.

    Required Keys:

        - img
        - bbox_center
        - bbox_scale
        - bbox_rotation (optional)
        - keypoints (optional)

    Modified Keys:

        - img
        - bbox_scale

    Added Keys:

        - input_size
        - transformed_keypoints

    Args:
        input_size (Tuple[int, int]): The input image size of the model in
            [w, h]. The bbox region will be cropped and resize to `input_size`
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 use_udp: bool = False) -> None:
        super().__init__()

        assert is_seq_of(input_size, int) and len(input_size) == 2, (
            f'Invalid input_size {input_size}')

        self.input_size = input_size
        self.use_udp = use_udp

    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
        """Reshape the bbox to a fixed aspect ratio.

        Args:
            bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.darray: The reshaped bbox scales in (n, 2)
        """

        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                              np.hstack([w, w / aspect_ratio]),
                              np.hstack([h * aspect_ratio, h]))
        return bbox_scale

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`TopdownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        # img = results['img'] // 4
        # for [x, y, _] in results['raw_ann_info']['true_points']:
        #     img = cv2.circle(img, center=(x, y), color=(0, 0, 255), thickness=-1, radius=3)
        # for [x1, y1, _, x2, y2, _, _] in results['raw_ann_info']['true_skeletons']:
        #     img = cv2.line(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=1)
        #
        # cv2.imwrite(r"E:\LJW\Git\mmpose\tools\0_LJW_tools\test_org.jpg", img)
        w, h = self.input_size
        warp_size = (int(w), int(h))

        # reshape bbox to fixed aspect ratio
        results['bbox_scale'] = self._fix_aspect_ratio(
            results['bbox_scale'], aspect_ratio=w / h)

        # TODO: support multi-instance
        assert results['bbox_center'].shape[0] == 1, (
            'Top-down heatmap only supports single instance. Got invalid '
            f'shape of bbox_center {results["bbox_center"].shape}.')

        center = results['bbox_center'][0]
        scale = results['bbox_scale'][0]
        if 'bbox_rotation' in results:
            rot = results['bbox_rotation'][0]
        else:
            rot = 0.

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                center, scale, rot, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

        if isinstance(results['img'], list):
            results['img'] = [
                cv2.warpAffine(
                    img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                for img in results['img']
            ]
        else:
            # 图像变换
            results['img'] = cv2.warpAffine(results['img'], warp_mat, warp_size, flags=cv2.INTER_LINEAR)

        if results.get('keypoints', None) is not None:
            # 关键点变换
            transformed_keypoints = results['keypoints'].copy()
            # Only transform (x, y) coordinates
            transformed_keypoints[..., :2] = cv2.transform(results['keypoints'][..., :2], warp_mat)

            # 真实标注关键点变化
            results['transformed_keypoints'] = transformed_keypoints
            true_points = np.array([results['raw_ann_info']['true_points']], dtype=np.float32)
            true_points[..., :2] = cv2.transform(true_points[..., :2], warp_mat)
            results['raw_ann_info']['true_points'] = true_points.astype(np.int32)[0].tolist()
            true_skeletons = np.array([results['raw_ann_info']['true_skeletons']], dtype=np.float32)
            true_skeletons[..., :2] = cv2.transform(true_skeletons[..., :2], warp_mat)
            true_skeletons[..., 3:5] = cv2.transform(true_skeletons[..., 3:5], warp_mat)
            results['raw_ann_info']['true_skeletons'] = true_skeletons.astype(np.int32)[0].tolist()
        results['input_size'] = (w, h)
        results['input_center'] = center
        # img = results['img'] // 4
        # for [x, y, _] in results['raw_ann_info']['true_points']:
        #     img = cv2.circle(img, center=(x, y), color=(0, 0, 255), thickness=-1, radius=3)
        # for [x1, y1, _, x2, y2, _, _] in results['raw_ann_info']['true_skeletons']:
        #     img = cv2.line(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=1)
        # cv2.imwrite(r"E:\LJW\Git\mmpose\tools\0_LJW_tools\test_affined.jpg", img)
        # exit()
        results['input_scale'] = scale  # scale = max(bbox_w, bbox_h) * padding

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'use_udp={self.use_udp})'
        return repr_str


@TRANSFORMS.register_module()
class LJW_TopdownAffine_2(BaseTransform):
    """Get the bbox image as the model input by affine transform.

    Required Keys:

        - img
        - bbox_center
        - bbox_scale
        - bbox_rotation (optional)
        - keypoints (optional)

    Modified Keys:

        - img
        - bbox_scale

    Added Keys:

        - input_size
        - transformed_keypoints

    Args:
        input_size (Tuple[int, int]): The input image size of the model in
            [w, h]. The bbox region will be cropped and resize to `input_size`
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 use_udp: bool = False) -> None:
        super().__init__()

        assert is_seq_of(input_size, int) and len(input_size) == 2, (
            f'Invalid input_size {input_size}')

        self.input_size = input_size
        self.use_udp = use_udp

    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
        """Reshape the bbox to a fixed aspect ratio.

        Args:
            bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.darray: The reshaped bbox scales in (n, 2)
        """

        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                              np.hstack([w, w / aspect_ratio]),
                              np.hstack([h * aspect_ratio, h]))
        return bbox_scale

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`TopdownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        # img = results['img'] // 4
        # for [x, y, _] in results['raw_ann_info']['true_points']:
        #     img = cv2.circle(img, center=(x, y), color=(0, 0, 255), thickness=-1, radius=3)
        # for [x1, y1, _, x2, y2, _, _] in results['raw_ann_info']['true_skeletons']:
        #     img = cv2.line(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=1)
        #
        # cv2.imwrite(r"E:\LJW\Git\mmpose\tools\0_LJW_tools\test_org.jpg", img)
        w, h = self.input_size
        warp_size = (int(w), int(h))

        # reshape bbox to fixed aspect ratio
        results['bbox_scale'] = self._fix_aspect_ratio(
            results['bbox_scale'], aspect_ratio=w / h)

        # TODO: support multi-instance
        assert results['bbox_center'].shape[0] == 1, (
            'Top-down heatmap only supports single instance. Got invalid '
            f'shape of bbox_center {results["bbox_center"].shape}.')

        center = results['bbox_center'][0]
        scale = results['bbox_scale'][0]
        if 'bbox_rotation' in results:
            rot = results['bbox_rotation'][0]
        else:
            rot = 0.

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                center, scale, rot, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

        if isinstance(results['img'], list):
            results['img'] = [
                cv2.warpAffine(
                    img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                for img in results['img']
            ]
        else:
            # 图像变换
            results['img'] = cv2.warpAffine(results['img'], warp_mat, warp_size, flags=cv2.INTER_LINEAR)

        if results.get('keypoints', None) is not None:
            # 关键点变换
            transformed_keypoints = results['keypoints'].copy()
            # Only transform (x, y) coordinates
            transformed_keypoints[..., :2] = cv2.transform(results['keypoints'][..., :2], warp_mat)
            results['transformed_keypoints'] = transformed_keypoints
            # 真实标注关键点变化
            for (key_name, temp_dict) in results['raw_ann_info'].items():
                if 'true_points' in key_name:
                    true_points = np.array([results['raw_ann_info'][key_name]], dtype=np.float32)
                    true_points[..., :2] = cv2.transform(true_points[..., :2], warp_mat)
                    results['raw_ann_info'][key_name] = true_points.astype(np.int32)[0].tolist()
                if 'true_skeletons' in key_name:
                    true_skeletons = np.array([results['raw_ann_info'][key_name]], dtype=np.float32)
                    true_skeletons[..., :2] = cv2.transform(true_skeletons[..., :2], warp_mat)
                    true_skeletons[..., 3:5] = cv2.transform(true_skeletons[..., 3:5], warp_mat)
                    results['raw_ann_info'][key_name] = true_skeletons.astype(np.int32)[0].tolist()
        results['input_size'] = (w, h)
        results['input_center'] = center
        # img = results['img'] // 4
        # for [x, y, _] in results['raw_ann_info']['true_points']:
        #     img = cv2.circle(img, center=(x, y), color=(0, 0, 255), thickness=-1, radius=3)
        # for [x1, y1, _, x2, y2, _, _] in results['raw_ann_info']['true_skeletons']:
        #     img = cv2.line(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=1)
        # cv2.imwrite(r"E:\LJW\Git\mmpose\tools\0_LJW_tools\test_affined.jpg", img)
        # exit()
        results['input_scale'] = scale  # scale = max(bbox_w, bbox_h) * padding

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'use_udp={self.use_udp})'
        return repr_str

