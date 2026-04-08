#!/bin/bash
#SBATCH --job-name=rf_mnist
#SBATCH --output=sample.out
#SBATCH --error=sample.out
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=xgpu
#SBATCH --gres=gpu:tesla:1

# Evita warning sul stack size imposta memoria stack
ulimit -s 10240

# Carica CUDA
module load cuda/11.5

# Attiva ambiente micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate py310

# Vai nella cartella del progetto
cd ~/variational_hrf

# Lancia il training
python sample.py \
        --output_dir='./results' \
        --exp_name='mnist_baseline' \
        --gpu=0 \
        --integration_method='euler' \
        --num_samples=10000 \
        --single_images=True \