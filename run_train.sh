#!/bin/bash

python train.py \
--output_dir='./output_dir' \
--exp_name='primo' \
--dataset='cifar10' \
--hrf=True \
--gpu=-1 \
--num_channel=128 \
--channel_mult=1,2 \
--total_steps=1000 \
--warmup=500 \
--batch_size=64 \
--ot_bs=64 \
--num_workers=2 \
--ema_decay=0.9999 \
--continue_train=False \
--save_step=10000 \
--tb_step=25
