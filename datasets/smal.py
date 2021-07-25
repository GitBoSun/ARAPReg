import os
import os.path as osp
from glob import glob

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from utils.read import read_mesh

from tqdm import tqdm
import numpy as np

class SMAL(InMemoryDataset):

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 pre_transform=None):
        if not osp.exists(osp.join(root, 'processed')):
            os.makedirs(osp.join(root, 'processed'))
        
        super().__init__(root, transform, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return 'smal.zip'
    
    @property
    def processed_file_names(self):
        return [ 'training.pt', 'test.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please '
            'move {} to {}'.format(self.raw_file_names, self.raw_dir))

    def process(self):
        print('Processing...')
        fps = glob(osp.join(self.raw_dir, '*/*.ply'))
        if len(fps) == 0:
            extract_zip(self.raw_paths[0], self.raw_dir, log=False)
            fps = glob(osp.join(self.raw_dir, '*/*.ply'))
        train_data_list, test_data_list = [], []
        train_id = 0
        val_id = 0
        
        poses = np.load(osp.join(self.raw_dir, 'results_pose_800/poses.npy'))
        
        for i in range(poses.shape[0]):
            print('processing %d/%d'%(i, poses.shape[0]))

            fp = osp.join(self.raw_dir, 'results_pose_800/%d.ply'%(i+1))
            pose_id = int(fp.split('/')[-1].split('.')[0]) - 1
            
            # 300 train, 100 test 
            if  pose_id> 299 and pose_id<400:
                data_id = val_id
                val_id = val_id + 1
            elif pose_id < 300:
                data_id = train_id
                train_id = train_id + 1
            
            data = read_mesh(fp, data_id, pose=poses[pose_id], return_face=False)
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if pose_id > 299 and pose_id < 400:
                test_data_list.append(data)
              
            elif pose_id < 300:
                train_data_list.append(data)

        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(test_data_list), self.processed_paths[1])

