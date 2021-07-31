#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python -W ignore main.py \
--exp_name 'model0' \
--device_idx 0 \
--batch_size 8 \
--epochs 2000 \
--lr 0.01 \
--arap_weight 0.05 \
--use_arap_epoch 800 \
--arap_eig_k 30 \
--decay_step 3 \
--latent_channels 96 \
--use_pose_init True \
--work_dir ./work_dir/SMAL \
--dataset SMAL \
--data_dir ./data/SMAL \
