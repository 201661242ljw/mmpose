# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from argparse import ArgumentParser

from mmcv.image import imread
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file', default=r"C:\Users\concrete\Desktop\0000_SCI\test\2.png")
    parser.add_argument('--config', help='Config file', default=config_path)
    parser.add_argument('--checkpoint', help='Checkpoint file', default=model_path)
    parser.add_argument('--out-file', help='Path to output file', default=out_path)
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        help='Visualize the predicted heatmap')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    if args.draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    model = init_model(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=cfg_options)

    # init visualizer
    model.cfg.visualizer.radius = args.radius
    model.cfg.visualizer.alpha = args.alpha
    model.cfg.visualizer.line_width = args.thickness

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(
        model.dataset_meta, skeleton_style=args.skeleton_style)

    # inference a single image
    batch_results = inference_topdown(model, args.img)
    results = merge_data_samples(batch_results)

    # show the results
    img = imread(args.img, channel_order='rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        kpt_thr=args.kpt_thr,
        draw_heatmap=args.draw_heatmap,
        show_kpt_idx=args.show_kpt_idx,
        skeleton_style=args.skeleton_style,
        show=args.show,
        out_file=args.out_file)

    if args.out_file is not None:
        print_log(
            f'the output image has been saved at {args.out_file}',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    config_path = r"E:\LJW\Git\mmpose\tools\LJW_Log\2024-01-30_00-38-01\td-hm_hrnet-w48_8xb32-210e_tower_2.py"
    model_path = r"E:\LJW\Git\mmpose\tools\LJW_Log\2024-01-30_00-38-01\best_coco_AP_epoch_96.pth"
    out_path = r"E:\LJW\Git\mmpose\tools\0_LJW_tools\test"
    # img_dir = r"C:\Users\concrete\Desktop\0000_SCI\test"
    # for file_name in os.listdir(img_dir):
    #     img_path = os.path.join(img_dir, file_name)
    #     s = r"E:\LJW\Git\mmpose\openmmlab\python.exe E:\LJW\Git\mmpose\demo\image_demo.py --img_path {}".format(img_path)
    #     os.system(s)
    main()
