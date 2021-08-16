import os
import os.path as osp
from glob import glob

import torch
from torch_geometric.data import InMemoryDataset, extract_zip, Data
from torch_geometric.utils import to_undirected

import numpy as np

class DFaust(InMemoryDataset):

    def __init__(self,
                 root='data/dfaust/',
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
        return 'dfaust.zip'

    @property
    def processed_file_names(self):
        return [ 'training.pt', 'test.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please '
            'move {} to {}'.format(self.raw_file_names, self.raw_dir))

    def process(self):
        
        def convert_to_data(face, points, data_id, pose=None):
            face = torch.from_numpy(faces).T.type(torch.long)
            x = torch.tensor(points, dtype=torch.float)
            edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
            edge_index = to_undirected(edge_index)
            if pose is not None:
                return Data(x=x, edge_index=edge_index, data_id=data_id, pose=pose)
            return Data(x=x, edge_index=edge_index, data_id=data_id)
        
        if not osp.exists(osp.join(self.raw_dir, 'train.npy')):
            extract_zip(self.raw_paths[0], self.raw_dir, log=False)

        train_points = np.load(osp.join(self.raw_dir, 'train.npy')).astype(
            np.float32).reshape((-1, 6890, 3))
        
        test_points = np.load(osp.join(self.raw_dir, 'test.npy')).astype(
            np.float32).reshape((-1, 6890, 3))
        
        #eval_points = np.load(osp.join(self.raw_dir, 'eval.npy')).astype(
        #    np.float32).reshape((-1, 6890, 3))
        
        faces = np.load(osp.join(self.raw_dir, 'faces.npy')).astype(
            np.int32).reshape((-1, 3))

        print('Processing...')
        
        train_data_list = []
        for i in range(train_points.shape[0]):
            data_id = i
            train = train_points[i]
            if i%100==0:
                print('processing training data %d/%d'%(i, train_points.shape[0]))

            data = convert_to_data(faces, train, data_id)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            train_data_list.append(data)
        torch.save(self.collate(train_data_list), self.processed_paths[0])
        
        
        """
        eval_data_list = []
        for data_id, eval in enumerate(tqdm(eval_points)):
            data = convert_to_data(faces, eval, data_id)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            eval_data_list.append(data)
        torch.save(self.collate(eval_data_list), self.processed_paths[1])
        """

        test_data_list = []
        data_id = 0
        for i in range(test_points.shape[0]):
            print('processing testing data %d/%d'%(i, test_points.shape[0]))
            test = test_points[i]

            data = convert_to_data(faces, test, data_id)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            test_data_list.append(data)
            data_id += 1 
        torch.save(self.collate(test_data_list), self.processed_paths[1])


if __name__ == '__main__':
    dfaust = DFaust()

