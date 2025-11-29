# ...existing code...
_base_ = 'mmdet::cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py'

# ====== 必需：导入自定义模块（MixStyle 实现）=====
custom_imports = dict(imports=['mmdet_custom.mixstyle_backbone'], allow_failed_import=False)

# ====== 必备：默认 scope ======
default_scope = 'mmdet'

# ====== 数据集类别 ======
classes = ('ship',)
metainfo = dict(classes=classes)

# ====== 路径（你只需要改这两个）======
# 注意：这里我故意不加结尾的斜杠，后面用 f'{data_root}/annotations/xxx.json' 来拼
data_root_hrsid = r'D:\Python\domainshipdataset\HRSID_png'
data_root_ssdd  = r'D:\Python\domainshipdataset\SSDD'

# 你的目录里必须满足：
# data_root_hrsid/
#   images/
#   annotations/train.json
#   annotations/val.json
# data_root_ssdd/
#   images/
#   annotations/test.json

# ====== 模型：改成 1 类（ship），并使用 MixStyleResNet 作为 backbone ======
model = dict(
    backbone=dict(
        type='MixStyleResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        mixstyle_cfg=dict(
            p=0.9,      # 激进：0.9
            alpha=0.5,  # 激进：0.5
            apply_layers=(0, 1, 2)
        )
    ),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False
            ),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False
            ),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False
            )
        ]
    )
)

# ====== 训练数据增强（先用稳妥的版本，去掉旋转）======
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=[(1333, 640), (1333, 800)],
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(640, 640),
        allow_negative_crop=True),
    dict(
        type='RandomFlip',
        prob=0.5,
        direction=['horizontal', 'vertical']),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs'),
]

# 验证 / 测试不要做花里胡哨增强，只 resize
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs'),
]

# ====== 训练 dataloader：HRSID （源域）======
train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root_hrsid,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=16),
        pipeline=train_pipeline,
    )
)

# ====== 源域验证（HRSID val）======
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root_hrsid,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
    )
)

# ====== 目标域测试（SSDD test：未见新域）======
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root_ssdd,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
    )
)

# ====== 评估器：源域 & 目标域 ======
val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root_hrsid}/annotations/val.json',
    metric='bbox',
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root_ssdd}/annotations/test.json',
    metric='bbox',
)

# ====== 训练策略（epoch / lr 等）======
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=120,  # 增加到 120
    val_interval=1,
)

# 优化器（继承 base 的 SGD，稍微调一下 lr 可按显存改）
optim_wrapper = dict(
    optimizer=dict(
        lr=0.01,    # 若 batch_size=2 觉得不稳，可以改成 0.005
        momentum=0.9,
        weight_decay=0.0001,
    )
)

# 调整学习率
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        begin=0,
        end=48,
        milestones=[32, 44],  # 改为 32, 44
        gamma=0.1),
]

# 日志 & 可视化（用默认就行）
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3)
)

# ====== 必须有 work_dir（你用 Runner.from_cfg 时会用到）======
work_dir = './work_dirs/hrsid_to_ssdd_cascade_r50'
# ...existing code...