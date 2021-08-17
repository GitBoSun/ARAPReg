#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore main.py \
--out_channels 32 32 32 64 \
--ds_factors 2 2 2 2 \
--exp_name 'arap' \
--device_idx 0 \
--batch_size 64 \
--epochs 450 \
--lr 1e-4 \
--test_lr 1e-3 \
--test_decay_step 5 \
--arap_weight 5e-4 \
--use_arap_epoch 150 \
--decay_step 3 \
--latent_channels 72 \
--use_vert_pca True \
--work_dir ./work_dir/DFaust \
--dataset DFaust \
--data_dir ./data/DFaust \
--distributed \
--alsotest \
#--continue_train True \
#--checkpoint work_dir/DFaust/out/arap/checkpoints/checkpoint_0410.pt \
#--test_checkpoint work_dir/DFaust/out/arap/test_checkpoints/checkpoint_01200.pt \
