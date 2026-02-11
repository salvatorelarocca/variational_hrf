#!/bin/bash

python ./variational_hrf/train.py \
--output_dir='./smoke_output' \
--exp_name='smoke_cirfar10_hrf' \
--dataset='cifar10' \
--model='for_cifar10mini' \
--hrf=True \
--gpu=0 \
--num_channel=128 \
--channel_mult=1,2,2,2 \
--total_steps=5001 \
--warmup=50 \
--batch_size=32 \
--ot_bs=16 \
--num_workers=0 \
--ema_decay=0.99 \
--continue_train=False \
--save_step=2500 \
--tb_step=50 \
--variational=True \
--latent_dim=128 \
--beta=0.5