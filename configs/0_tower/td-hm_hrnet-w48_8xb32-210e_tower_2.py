_base_ = ['../../configs/_base_/default_runtime.py']

heatmap_scale = [16, 8, 4]

batch_size = 2

input_size = 1024
epoch_num = 100

use_medium_satge = True
target_form = 3
# --------------------------------
# --------------------------------
output_form_1 = True
sigma_1 = 2
paf_half_width_1 = 2.5
num_keypoints_1 = 5
num_skeletons_1 = 5
# --------------------------------
# --------------------------------
output_form_2 = True
sigma_2 = 2
paf_half_width_2 = 3
num_keypoints_2 = 5
num_skeletons_2 = 5
# --------------------------------
# --------------------------------
output_form_3 = True
sigma_3 = 1.5
paf_half_width_3 = 2
num_keypoints_3 = 14
num_skeletons_3 = 37
# --------------------------------
# --------------------------------
channel_labels = [
    [num_keypoints_1, num_skeletons_1, int(output_form_1), sigma_1],
    [num_keypoints_2, num_skeletons_2, int(output_form_2), sigma_2],
    [num_keypoints_3, num_skeletons_3, int(output_form_3), sigma_3],
]
if use_medium_satge:
    num_out_channel = ((num_keypoints_1 + num_skeletons_1 * 2) * int(output_form_1) +
                       (num_keypoints_2 + num_skeletons_2 * 2) * int(output_form_2) +
                       (num_keypoints_3 + num_skeletons_3 * 2) * int(output_form_3))

else:
    num_out_channel = [num_keypoints_1 + num_skeletons_1 * 2, num_keypoints_2 + num_skeletons_2 * 2,
                       num_keypoints_3 + num_skeletons_3 * 2][target_form - 1]
    channel_labels = [channel_labels[target_form - 1]]
    channel_labels[0][2] = 1
# runtime
train_cfg = dict(max_epochs=epoch_num, val_interval=1)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
# codec = dict(
#     type='MSRAHeatmap', input_size=(input_size, input_size), heatmap_size=(input_size // 4, input_size // 4),
#     sigma=sigma)

codec = dict(
    type='LJWHeatmapAndPaf_2',
    input_size=(input_size, input_size),
    heatmap_size=(
        input_size // heatmap_scale[0],
        input_size // heatmap_scale[0],
        input_size // heatmap_scale[1],
        input_size // heatmap_scale[1],
        input_size // heatmap_scale[2],
        input_size // heatmap_scale[2]
    ),
    heatmap_scale=heatmap_scale,
    use_medium_satge=use_medium_satge,
    target_form=target_form,

    output_form_1=output_form_1,
    sigma_1=sigma_1,
    paf_half_width_1=paf_half_width_1,
    num_keypoints_1=num_keypoints_1,
    num_skeletons_1=num_skeletons_1,

    output_form_2=output_form_2,
    sigma_2=sigma_2,
    paf_half_width_2=paf_half_width_2,
    num_keypoints_2=num_keypoints_2,
    num_skeletons_2=num_skeletons_2,

    output_form_3=output_form_3,
    sigma_3=sigma_3,
    paf_half_width_3=paf_half_width_3,
    num_keypoints_3=num_keypoints_3,
    num_skeletons_3=num_skeletons_3
)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384),
                multiscale_output=True)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w48-8ef0771d.pth',
            # checkpoint=''
        ),
    ),
    # backbone=dict(
    #     type='HRNet_LJW',
    #     in_channels=3,
    #     use_medium_satge=use_medium_satge,
    #     output_form_1=output_form_1,
    #     output_form_2=output_form_2,
    #     output_form_3=output_form_3,
    #     num_keypoints_1=num_keypoints_1,
    #     num_skeletons_1=num_skeletons_1,
    #     num_keypoints_2=num_keypoints_2,
    #     num_skeletons_2=num_skeletons_2,
    #     num_keypoints_3=num_keypoints_3,
    #     num_skeletons_3=num_skeletons_3,
    #     extra=dict(
    #         stage1=dict(
    #             num_modules=1,
    #             num_branches=1,
    #             block='BOTTLENECK',
    #             num_blocks=(4,),
    #             num_channels=(64,)),
    #         stage2=dict(
    #             num_modules=1,
    #             num_branches=2,
    #             block='BASIC',
    #             num_blocks=(4, 4),
    #             num_channels=(48, 96)),
    #         stage3=dict(
    #             num_modules=4,
    #             num_branches=3,
    #             block='BASIC',
    #             num_blocks=(4, 4, 4),
    #             num_channels=(48, 96, 192)),
    #         stage4=dict(
    #             num_modules=3,
    #             num_branches=4,
    #             block='BASIC',
    #             num_blocks=(4, 4, 4, 4),
    #             num_channels=(48, 96, 192, 384),
    #             multiscale_output=True)),
    #     init_cfg=dict(
    #         type='Pretrained',
    #         checkpoint='https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w48-8ef0771d.pth'),
    #     # checkpoint=r'E:\LJW\Git\mmpose\tools\LJW_Log\2023-12-03_00-22-42\epoch_210.pth'),
    # ),
    head=dict(
        type='LJW_HeatmapHead',
        use_medium_satge=use_medium_satge,
        target_form=target_form,
        heatmap_scale=heatmap_scale,
        output_form_1=output_form_1,
        output_form_2=output_form_2,
        output_form_3=output_form_3,
        channel_labels=channel_labels,
        in_channels=48,
        out_channels=num_out_channel,
        deconv_out_channels=None,
        loss=dict(type='LJW_KeypointMSELoss', use_target_weight=False),
        decoder=codec),
    test_cfg=dict(
        flip_test=False,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'TowerDataset'
data_mode = 'topdown'
data_root = r'data/00_Tower_Dataset/{}/'.format(input_size)

# pipelines
train_pipeline = [
    dict(type='LJW_LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.1),  # results['bbox_center'] = center    # results['bbox_scale'] = scale
    dict(type='LJW_RandomFlip_2', direction='horizontal', prob=0.5),
    # results['bbox'],#results['bbox_center'] # results['keypoints'] = keypoints    #results['keypoints_visible'] = keypoints_visible
    # dict(type='RandomHalfBody'),
    dict(type='LJW_RandomBBoxTransform',
         shift_factor=0.1, shift_prob=1.0,
         scale_factor=(0.95, 1.15), scale_prob=1.0,
         rotate_factor=7.5, rotate_prob=1.0),
    dict(type='LJW_TopdownAffine_2', input_size=codec['input_size']),
    # results['input_size'] = (w, h)       # results['input_center'] = center     #   results['input_scale'] = scale
    dict(type='LJW_GenerateTarget_2', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LJW_LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.05),
    dict(type='LJW_TopdownAffine_2', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='anns/tower_keypoints_train_2.json',
        data_prefix=dict(img='imgs/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='anns/tower_keypoints_val_2.json',
        # bbox_file='data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='imgs/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='anns/tower_keypoints_test_2.json',
        # bbox_file='data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='imgs/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

# evaluators
# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'anns/tower_keypoints_val_2.json')
# test_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'anns/tower_keypoints_test_2.json')

val_evaluator = dict(
    type='TowerMetric',
    sigma=[sigma_1,sigma_2,sigma_3],
    ann_file=data_root + 'anns/tower_keypoints_val_2.json')
test_evaluator = dict(
    type='TowerMetric',
    sigma=[sigma_1,sigma_2,sigma_3],
    ann_file=data_root + 'anns/tower_keypoints_test_2.json')
