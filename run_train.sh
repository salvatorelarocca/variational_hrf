#!/bin/bash
#num_channels / 4 deve essere divisibile per 32 numero dei gruppi di normalizzazione presenti in nn.py
#successivamente num_channels / 4 deve essere divisibile per num_heads_channels
#B C H W per ogni batch si dividono i canali in gruppi si effettua una normalizzazione per ogni gruppo separatamente
python train.py \
--output_dir='./smoke_output' \
--exp_name='smoke_cirfar10_hrf' \
--dataset='cifar10' \
--model='for_cifar10mini' \
--hrf=True \
--gpu=-1 \
--num_channel=128 \
--channel_mult=1,2 \
--total_steps=10 \
--warmup=50 \
--batch_size=16 \
--ot_bs=16 \
--num_workers=0 \
--ema_decay=0.99 \
--continue_train=False \
--save_step=9 \
--tb_step=0 \
--variational=True \
--latent_dim=16 \
--beta=0.5 \
--use_scale_shift_norm=False
