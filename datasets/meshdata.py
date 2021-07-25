import openmesh as om
from datasets import SMAL, DFaust, Bone
from sklearn.decomposition import PCA
import numpy as np
import sys 
sys.path.append("./")
import torch 

class MeshData(object):
    def __init__(self,
                 root,
                 template_fp,
                 dataset='DFaust',
                 transform=None,
                 pre_transform=None,
                 pca_n_comp=8,
                 vert_pca=False,
                 heat_kernel=False):
        self.root = root
        self.template_fp = template_fp
        self.dataset = dataset
        self.transform = transform
        self.pre_transform = pre_transform
        self.train_dataset = None
        self.test_dataste = None
        self.template_points = None
        self.template_face = None
        self.mean = None
        self.std = None
        self.num_nodes = None
        self.use_vert_pca = vert_pca
        self.use_heat_kernel = heat_kernel
        self.pca = PCA(n_components=pca_n_comp)

        self.load()

    def load(self):
        if self.dataset=='SMAL':
            self.train_dataset = SMAL(self.root,
                                  train=True,
                                  transform=self.transform,
                                  pre_transform=self.pre_transform)
            self.test_dataset = SMAL(self.root,
                                 train=False,
                                 transform=self.transform,
                                 pre_transform=self.pre_transform)

        elif self.dataset=='DFaust':
            self.train_dataset = DFaust(self.root,
                                  train=True,
                                  transform=self.transform,
                                  pre_transform=self.pre_transform)
            self.test_dataset = DFaust(self.root,
                                 train=False,
                                 transform=self.transform,
                                 pre_transform=self.pre_transform)
        elif self.dataset=='Bone':
            self.train_dataset = Bone(self.root,
                                      train=True,
                                      transform=self.transform,
                                      pre_transform=self.pre_transform)
            self.test_dataset = Bone(self.root,
                                     train=False,
                                     transform=self.transform,
                                     pre_transform=self.pre_transform)

        tmp_mesh = om.read_trimesh(self.template_fp)
        self.template_points = tmp_mesh.points()
        self.template_face = tmp_mesh.face_vertex_indices()
        self.num_nodes = self.train_dataset[0].num_nodes

        self.num_train_graph = len(self.train_dataset)
        self.num_test_graph = len(self.test_dataset)
        
        self.mean = self.train_dataset.data.x.view(self.num_train_graph, -1,
                                                   3).mean(dim=0)
        self.std = self.train_dataset.data.x.view(self.num_train_graph, -1,
                                                  3).std(dim=0)
        if self.dataset=='SMAL':
            self.mean = torch.zeros(self.mean.shape)
            self.std = torch.ones(self.std.shape)
        
        self.normalize()

    def normalize(self):

        vertices_train = self.train_dataset.data.x.view(self.num_train_graph, -1, 3).numpy()
        vertices_test = self.test_dataset.data.x.view(self.num_test_graph, -1, 3).numpy()
        
        print('Normalizing...')
        self.train_dataset.data.x = (
            (self.train_dataset.data.x.view(self.num_train_graph, -1, 3) -
             self.mean) / self.std).view(-1, 3)
        self.test_dataset.data.x = (
            (self.test_dataset.data.x.view(self.num_test_graph, -1, 3) -
             self.mean) / self.std).view(-1, 3)
        
        if self.use_vert_pca:
            print("Computing vertex PCA...")
            self.pca.fit(np.reshape(vertices_train, (self.num_train_graph, -1)))
            #train_pca_sv = self.pca.singular_values_
            pca_axes = self.pca.components_
            train_pca_sv= np.matmul(np.reshape(vertices_train, (self.num_train_graph, -1)), pca_axes.transpose())
            test_pca_sv = np.matmul(np.reshape(vertices_test, (self.num_test_graph, -1)), pca_axes.transpose())
            print('train pca', train_pca_sv.mean(),np.std(train_pca_sv, axis=0), train_pca_sv.max(), train_pca_sv.min())
            # print('test pca', test_pca_sv.mean(),np.std(test_pca_sv, axis=0), test_pca_sv.max(), test_pca_sv.min())
            pca_sv_mean = np.mean(train_pca_sv, axis=0)
            pca_sv_std = np.std(train_pca_sv, axis=0)
            self.train_pca_sv = (train_pca_sv - pca_sv_mean)/pca_sv_std
            self.test_pca_sv = (test_pca_sv - pca_sv_mean)/pca_sv_std

        print('Done!')

    def save_mesh(self, fp, x):
        x = x * self.std + self.mean
        om.write_mesh(fp, om.TriMesh(x.numpy(), self.template_face))
