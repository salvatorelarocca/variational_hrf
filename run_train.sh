#!/bin/bash
#num_channels / 4 deve essere divisibile per 32 numero dei gruppi di normalizzazione presenti in nn.py
#successivamente num_channels / 4 deve essere divisibile per num_heads_channels
#B C H W per ogni batch si dividono i canali in gruppi si effettua una normalizzazione per ogni gruppo separatamente
#utilizzato se FM=ExactOptimalTransportConditionalFlowMatch
python train.py \
    --output_dir='./test' \
    --exp_name='exp_vegeta' \
    --dataset='mnist' \
    --model_type='unet_cat_hrf' \
    --variational=True \
    --latent_dim=128 \
    --beta=0.5 \
    --integration_method="euler" \
    --gpu=-1 \
    --num_channel=128 \
    --channel_mult=1 \
    --total_steps=10 \
    --warmup=50 \
    --batch_size=16 \
    --ot_bs=16 \
    --num_workers=0 \
    --ema_decay=0.99 \
    --continue_train=True \
    --save_step=9 \
    --tb_step=1 \
    --use_scale_shift_norm=False \