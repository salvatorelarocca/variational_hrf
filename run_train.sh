#!/bin/bash
#num_channels / 4 deve essere divisibile per 32 numero dei gruppi di normalizzazione presenti in nn.py
#successivamente num_channels / 4 deve essere divisibile per num_heads_channels
#B C H W per ogni batch si dividono i canali in gruppi si effettua una normalizzazione per ogni gruppo separatamente
#utilizzato se FM=ExactOptimalTransportConditionalFlowMatch
python train.py \
    --output_dir='./test' \
    --exp_name='prova2' \
    --dataset='mnist' \
    --model_type='unet_cat_hrf' \
    --variational=True \
    --latent_dim=16 \
    --beta=0.3 \
    --kl_warmup_frac=0.6 \
    --free_bits=1.0 \
    --n_cycles=2 \
    --beta_schedule="cyclic_cosine" \
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
    --generate_samples=False