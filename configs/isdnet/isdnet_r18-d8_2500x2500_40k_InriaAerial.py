
_base_ = [
    '../_base_/models/isdnet_r50-d8.py', '../_base_/datasets/aerial_2500x2500.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained=None,
    down_ratio=4,
    backbone=dict(
        type='ResNetV1c',
        depth=18,
        # ... (other backbone settings)
        init_cfg=dict(type='Pretrained', checkpoint='/home/zzn/.cache/torch/hub/checkpoints/resnet18_v1c-b5776b93.pth') # <-- ADD THIS
    ),
    decode_head=[
        dict(
            type='RefineASPPHead',
            in_channels=512,
            in_index=3,
            channels=128,
            dilations=(1, 12, 24, 36),
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='ISDHead',
            in_channels=3,
            prev_channels=128,
            down_ratio=4,
            channels=128,
            num_classes=2,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False,
            # stdc_net=dict(
            #     pretrain_model='checkpoints/STDCNet813M_73.91.tar'),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=2))
