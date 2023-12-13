_base_ = ['../../configs/_base_/default_runtime.py']

batch_size = 4

input_size = 640
sigma = 3
paf_half_width = 2

num_keypoints = 5
num_skeletons = 5
heatmap_scale = 4

num_out_channel = num_keypoints + num_skeletons * 2

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

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
    type='LJWHeatmapAndPaf',
    input_size=(input_size, input_size),
    heatmap_size=(input_size // heatmap_scale, input_size // heatmap_scale),
    sigma=sigma,
    paf_half_width=paf_half_width,
    heatmap_scale=heatmap_scale,
    num_keypoints=num_keypoints,
    num_skeletons=num_skeletons
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
                num_channels=(48, 96, 192, 384))),
        # init_cfg=dict(
        #     type='Pretrained',
        #     # checkpoint='https://download.openmmlab.com/mmpose/'
        #     #            'pretrain_models/hrnet_w48-8ef0771d.pth',
        #     checkpoint=''
        # ),
    ),

    
    head=dict(
        type='HeatmapHead',
        in_channels=48,
        out_channels=num_out_channel,
        deconv_out_channels=None,
        loss=dict(type='LJW_KeypointMSELoss', use_target_weight=False),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'TowerDataset'
data_mode = 'topdown'
data_root = r'../00_Tower_Dataset/{}/'.format(input_size)

# pipelines
train_pipeline = [
    dict(type='LJW_LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.15),  # results['bbox_center'] = center    # results['bbox_scale'] = scale
    dict(type='LJW_RandomFlip', direction='horizontal', prob=0.5),
    # results['bbox'],#results['bbox_center'] # results['keypoints'] = keypoints    #results['keypoints_visible'] = keypoints_visible
    # dict(type='RandomHalfBody'),
    dict(type='LJW_RandomBBoxTransform',
         shift_factor=0.1, shift_prob=1.0,
         scale_factor=(1, 1.5), scale_prob=1.0,
         rotate_factor=10.0, rotate_prob=1.0),
    dict(type='LJW_TopdownAffine', input_size=codec['input_size']),
    # results['input_size'] = (w, h)       # results['input_center'] = center     #   results['input_scale'] = scale
    dict(type='LJW_GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LJW_LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.15),
    dict(type='LJW_TopdownAffine', input_size=codec['input_size']),
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
        ann_file='anns/tower_keypoints_train.json',
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
        ann_file='anns/tower_keypoints_val.json',
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
        ann_file='anns/tower_keypoints_test.json',
        # bbox_file='data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='imgs/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'anns/tower_keypoints_val.json')
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'anns/tower_keypoints_test.json')
