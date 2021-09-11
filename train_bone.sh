#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--exp_name 'model0' \
--device_idx 0 \
--epochs 4500 \
--lr 8e-3 \
--arap_weight 0.05 \
--use_arap_epoch 1000 \
--decay_step 10 \
--latent_channels 8 \
--use_vert_pca True \
--work_dir ./work_dir/bone/femur \
--dataset Bone \
--data_dir ./data/bone/femur \
