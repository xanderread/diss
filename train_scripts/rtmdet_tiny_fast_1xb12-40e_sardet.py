# this should be ran using the mmyolo 3.xx version - /mmyolo/tools/train.py

_base_ = 'rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py'

data_root = './data/SARDetDataset/' # change the path to the dataset
class_name = ('person', ) 
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

num_epochs_stage2 = 4

max_epochs = 100 
train_batch_size_per_gpu = 16
train_num_workers = 4
val_batch_size_per_gpu = 16
val_num_workers = 2

load_from = None

model = dict(
    backbone=dict(frozen_stages=0),
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        
        data_root=data_root,
        metainfo=metainfo,
        # Dataset annotation file of json path
        ann_file='annotations/train.json',
        # Dataset prefix
        data_prefix=dict(img='train/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/valid.json',
        data_prefix=dict(img='valid/')))


test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/')))



param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=_base_.lr_start_factor,
        by_epoch=False,
        begin=0,
        end=30),
    dict(
        type='CosineAnnealingLR',
        eta_min=_base_.base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

_base_.custom_hooks[1].switch_epoch = max_epochs - num_epochs_stage2

val_evaluator = dict(ann_file=data_root + 'annotations/valid.json')
test_evaluator = dict(type='mmdet.CocoMetric',ann_file=data_root + 'annotations/test.json',metric='bbox')


 
default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    logger=dict(type='LoggerHook', interval=1))
train_cfg = dict(max_epochs=max_epochs, val_interval=1)


visualizer = dict(
    vis_backends = [
        dict(type='LocalVisBackend'),  
        dict(
            type='WandbVisBackend',
           init_kwargs={
            'project': 'rtmdet',
            'group': 'maskrcnn-r50-fpn-1x-coco'
         },
       
        ),

    ]
)
