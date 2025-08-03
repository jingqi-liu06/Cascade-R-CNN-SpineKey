#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=6 \
    --master_port=29500 \
    /home/srt19/jingqi/X_ray_jingqi/ver_det/train_mmdet.py \
    /home/srt19/jingqi/X_ray_jingqi/ver_det/det_cascade-mask-rcnn.py \
    --work-dir /home/srt19/jingqi/X_ray_jingqi/ver_det/quanjizhui_cascade \
    --launcher pytorch