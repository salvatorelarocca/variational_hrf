#!/bin/bash

python train.py \
--output_dir='./smoke_output' \
--exp_name='smoke_cirfar10_hrf' \
--dataset='cifar10' \
--model='for_cifar10mini' \
--hrf=True \
--gpu=-1 \
--num_channel=128 \
--channel_mult=1,2 \
--total_steps=100 \
--warmup=50 \
--batch_size=1 \
--ot_bs=16 \
--num_workers=0 \
--ema_decay=0.99 \
--continue_train=False \
--save_step=20 \
--tb_step=0
