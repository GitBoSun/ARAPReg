import os
import os.path as osp
from glob import glob

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from utils.read import read_mesh

from tqdm import tqdm


class Bone(InMemoryDataset):
    url = 'Not needed'

    categories = [
        'femur',
        'pelvis',
        'scapula',
        'tibia',
    ]

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 pre_transform=None):
        if not osp.exists(osp.join(root, 'processed',)):
            os.makedirs(osp.join(root, 'processed',))

        super().__init__(root, transform, pre_transform)
        self.train_list = []
        self.test_list = []
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return 'bone_data.zip'

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def process(self):
        print('Processing...')
        
        fps = glob(osp.join(self.raw_dir, '*.stl'))
        if len(fps) == 0:
            extract_zip(self.raw_paths[0], self.raw_dir, log=False)
            fps = glob(osp.join(self.raw_dir, '*.stl'))
        train_data_list, test_data_list = [], []
        train_id = 0
        val_id = 0
        for idx, fp in enumerate(tqdm(fps)):
            if (idx % 100) < 10:
                data_id = val_id
                val_id = val_id + 1
            else:
                data_id = train_id
                train_id = train_id + 1
            data = read_mesh(fp, data_id)
            # data = read_mesh(fp)
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if (idx % 100) < 10:
                test_data_list.append(data)
            else:
                train_data_list.append(data)

        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(test_data_list), self.processed_paths[1])
