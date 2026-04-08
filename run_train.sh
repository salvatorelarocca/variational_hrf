#!/bin/bash

python train.py \
    --output_dir='./results' \
    --exp_name='prova6' \
    --dataset='mnist' \
    --model_type='unet_cat_hrf' \
    --variational=True \
    --beta=0.3 \
    --integration_method="euler" \
    --gpu=-1 \
    --num_channel=128 \
    --channel_mult=1 \
    --total_steps=31 \
    --warmup=50 \
    --batch_size=16 \
    --ot_bs=16 \
    --num_workers=0 \
    --ema_decay=0.99 \
    --continue_train=False \
    --save_step=4 \
    --tb_step=1 \
    --use_scale_shift_norm=False \
    --generate_samples=True \
    --val_batches=5