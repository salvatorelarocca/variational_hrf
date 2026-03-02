#!/bin/bash

python ./variational_hrf/train.py \
    --output_dir='./test' \
    --exp_name='exp_vegeta' \
    --dataset='mnist' \
    --model='unet_cat_hrf' \
    --hrf=True \
    --variational=True \
    --latent_dim=32 \
    --beta=0.5 \
    --kl_warmup_frac=0.2 \
    --free_bits=0.5 \
    --beta=0.5 \
    --gpu=0 \
    --num_channel=64 \
    --channel_mult=1,2,2 \
    --total_steps=50001 \
    --warmup=50 \
    --batch_size=32 \
    --ot_bs=16 \
    --num_workers=0 \
    --ema_decay=0.99 \
    --continue_train=False \
    --save_step=200 \
    --tb_step=10 \
    --use_scale_shift_norm=False \