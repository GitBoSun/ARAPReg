#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 python -W ignore main.py \
--exp_name 'final_arap0.05_1000_k0_normloss_1' \
--device_idx 0 \
--batch_size 8 \
--epochs 2000 \
--lr 0.01 \
--arap_weight 0.05 \
--use_arap_epoch 1000 \
--arap_eig_k 0 \
--decay_step 3 \
--latent_channels 96 \
--use_pose_init True \
--work_dir /scratch/cluster/bosun/decoder_coma_torch/smal_full_new \
--dataset SMAL \
--data_dir /scratch/cluster/bosun/decoder_coma_torch/smal_full_new/data_300 \
--continue_train True \
