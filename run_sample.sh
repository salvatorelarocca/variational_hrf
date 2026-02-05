#!/bin/bash

python sample.py \
--output_dir='./smoke_output' \
--exp_name='smoke_cirfar10_hrf' \
--dataset='cifar10' \
--model='for_cifar10mini' \
--hrf=True \
--gpu=-1 \
--num_channel=64 \
--channel_mult=1,2
