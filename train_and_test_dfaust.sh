#!/usr/bin/env bash
#GPUS=8
#PORT=${PORT:-61816}
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT main.py --launcher pytorch \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore main.py \
--out_channels 32 32 32 64 \
--ds_factors 2 2 2 2 \
--exp_name 'arap' \
--device_idx 0 \
--batch_size 64 \
--epochs 300 \
--lr 1e-4 \
--test_lr 1e-3 \
--test_decay_step 5 \
--arap_weight 5e-4 \
--use_arap_epoch 150 \
--arap_eig_k 30 \
--decay_step 3 \
--latent_channels 72 \
--use_vert_pca True \
--work_dir ./work_dir/DFaust \
--dataset DFaust \
--data_dir ./data/DFaust \
--distributed \
--alsotest \
--continue_train True \
#--checkpoint work_dir/DFaust/out/model0/checkpoints/checkpoint_0850.pt
