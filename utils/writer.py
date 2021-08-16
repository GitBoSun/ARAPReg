import os
import time
import torch
import json
from glob import glob


class Writer:
    def __init__(self, args=None):
        self.args = args

        if self.args is not None:
            tmp_log_list = glob(os.path.join(args.out_dir, 'log*'))
            
            self.test_log_file = os.path.join(
                args.out_dir, 'test_log_{:s}.txt'.format(
                    time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))
            if len(tmp_log_list) == 0:
                self.log_file = os.path.join(
                    args.out_dir, 'log_{:s}.txt'.format(
                        time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))
            else:
                self.log_file = tmp_log_list[0]
        message = '{}'.format(self.args)
        if args.mode=='test':
            with open(self.test_log_file, 'a') as log_file:
                log_file.write('{:s}\n'.format(message))
        else:
            with open(self.log_file, 'a') as log_file:
                log_file.write('{:s}\n'.format(message))
        

    def print_info(self, info):
        message = 'Epoch: {}/{}, Time: {:.3f}s, Train Loss: {:.5f}, L1: {:.5f}, arap: {:.5f} MSE: {:.5f}, lr: {:.6f}' \
                .format(info['current_epoch'], info['epochs'], info['t_duration'], \
                info['train_loss'], info['l1_loss'], info['arap_loss'],  info['mse_error'], info['lr']) 
        
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        #print(message)

    def print_info_test(self, info):
        message = '[test] Epoch: {}/{}, Time: {:.3f}s, Train Loss: {:.5f}, L1: {:.5f}, arap: {:.5f} MSE: {:.6f} lr: {:.6f}' \
                .format(info['test_current_epoch'], info['test_epochs'], info['t_duration'], \
                info['test_loss'], info['l1_loss'], info['arap_loss'],info['mse_error'],info['lr'])

        with open(self.test_log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        #print(message)

    def save_checkpoint(self, model, latent_vecs, optimizer, scheduler,
                        epoch, test=False):
        model_path = self.args.checkpoints_dir
        if test==True:
            model_path = self.args.checkpoints_dir_test
            print('save test checkpoint')

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_latent_vecs': latent_vecs.state_dict(),
            },
            os.path.join(
                model_path, 'checkpoint_{:04d}.pt'.format(epoch))
        )

    def load_checkpoint(self, model, latent_vecs=None, optimizer=None,
                        scheduler=None, test=False, checkpoint=None):
        model_path = self.args.checkpoints_dir
        if test==True:
            model_path = self.args.checkpoints_dir_test

        if checkpoint is None:
            checkpoint_list = glob(os.path.join(model_path, 'checkpoint_*'))
            if len(checkpoint_list)==0:
                print('train from scrach')
                return 0
            latest_checkpoint = sorted(checkpoint_list)[-1]
        else:
            latest_checkpoint = checkpoint
        print("loading model from ", latest_checkpoint)
        data = torch.load(latest_checkpoint)
        model.load_state_dict(data["model_state_dict"])
        if latent_vecs:
            latent_vecs.load_state_dict(data["train_latent_vecs"])
        if scheduler:
            scheduler.load_state_dict(data["scheduler_state_dict"])
        if optimizer:
            optimizer.load_state_dict(data["optimizer_state_dict"])
        print("loaded!")
        return data["epoch"]
