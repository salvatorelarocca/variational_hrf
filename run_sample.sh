#!/bin/bash

python sample.py \
  --output_dir='./test' \
  --exp_name='exp_vegeta' \
  --gpu=-1 \
  --integration_method='euler' \
  --num_samples=4