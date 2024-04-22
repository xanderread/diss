_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x_sar_det.py', '../_base_/default_runtime.py'
]

# this should be ran using the mmdet 3.xx version - /mmyolo/tools/train.py

data_root = './data/SARDetDataset/' # dataset root
class_name = ('person', )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])
model = dict(backbone=dict(frozen_stages=0, depth=50),roi_head=dict(bbox_head=dict(num_classes=1)))

max_epochs = 100 
train_batch_size_per_gpu = 8
train_num_workers = 4
val_batch_size_per_gpu = 8 
val_num_workers = 2


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



val_evaluator = dict(ann_file=data_root + 'annotations/valid.json')
test_evaluator = dict(ann_file=data_root + 'annotations/test.json')

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    logger=dict(type='LoggerHook', interval=1))




train_cfg = dict(max_epochs=max_epochs, val_interval=1)


visualizer = dict(
    vis_backends = [
        dict(type='LocalVisBackend'),  # For local visualizations
        dict(
            type='WandbVisBackend',
           init_kwargs={
            'project': 'rtmdet',
            'group': 'maskrcnn-r50-fpn-1x-coco'
         },
       
        ),

    ]
)
