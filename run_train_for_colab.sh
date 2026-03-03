#!/bin/bash

python ./variational_hrf/train.py \
    --output_dir='./test' \
    --exp_name='exp_goku' \
    --dataset='mnist' \
    --model_type='unet_cat_hrf' \
    --variational=True \
    --latent_dim=32 \
    --beta=0.5 \
    --kl_warmup_frac=0.3 \
    --free_bits=0.5 \
    --gpu=0 \
    --num_channel=128 \
    --channel_mult=1,2,2 \
    --total_steps=60001 \
    --warmup=5000 \
    --batch_size=128 \
    --ot_bs=16 \
    --num_workers=0 \
    --ema_decay=0.99 \
    --continue_train=False \
    --save_step=2000 \
    --tb_step=50 \
    --use_scale_shift_norm=False \