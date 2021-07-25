import pickle
import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T
from psbody.mesh import Mesh

from models import AE, Pool
from datasets import MeshData
from utils import utils, writer, train_eval, DataLoader, mesh_sampling

parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--exp_name', type=str, default='model0')
parser.add_argument('--dataset', type=str, default='DFAUST')
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--device_idx', type=int, default=0)
parser.add_argument('--cpu', action='store_true', help='if True, use CPU only')
parser.add_argument('--mode', type=str, default='train', help='[train, test, interpolate, extraplate]')
parser.add_argument('--work_dir', type=str, default='./out')
parser.add_argument('--data_dir', type=str, default='./data')

# network hyperparameters
parser.add_argument('--out_channels',
                    nargs='+',
                    #default=[32, 32, 32, 64],
                    default=[16, 16, 16, 32],
                    type=int)

parser.add_argument('--latent_channels', type=int, default=8)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--K', type=int, default=6)

# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=8e-3,)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--test_lr', type=float, default=0.01)
parser.add_argument('--test_decay_step', type=int, default=1)

parser.add_argument('--arap_weight', type=float, default=0.05)
parser.add_argument('--use_arap_epoch', type=int, default=600, help='epoch that we start to use arap loss')
parser.add_argument('--arap_eig_k', type=int, default=60, help='only care about last k eigvals of the hessian matrix in ARAP')

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--test_epochs', type=int, default=2000)
parser.add_argument('--continue_train', type=bool, default=False, help='If True, continue training from last checkpoint')


# interpolate
parser.add_argument('--inter_num', type=int, default=10, help='number of intermediate shapes between two shapes in interpolation')
parser.add_argument('--extra_num', type=int, default=5, help='number of extrapolation perturbations per shape')
parser.add_argument('--extra_thres', type=float, default=0.2)


# others
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use_vert_pca', type=bool, default=False, help='If True, use the vertex PCA as the latent vector initialization [DFAUST, Bone]')
parser.add_argument('--use_pose_init', type=bool, default=False, help='If True, use the provided pose vector as the latent vector initialization in training [SMAL]')

args = parser.parse_args()

args.data_fp =osp.join(args.data_dir)
args.out_dir = osp.join(args.work_dir, 'out', args.exp_name) # save checkpoints and logs 
args.results_dir = osp.join(args.work_dir, 'results', args.exp_name) # save training and testing results 
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
args.checkpoints_dir_test = osp.join(args.out_dir, 'test_checkpoints')
print(args)

utils.makedirs(args.out_dir)
utils.makedirs(args.checkpoints_dir)
utils.makedirs(args.checkpoints_dir_test)
utils.makedirs(args.results_dir)
results_dir_train = os.path.join(args.results_dir, "train")
results_dir_test = os.path.join(args.results_dir, "test")
utils.makedirs(results_dir_train)
utils.makedirs(results_dir_test)
writer = writer.Writer(args)

if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda', args.device_idx)


torch.set_num_threads(args.n_threads)

# deterministic
torch.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

# load dataset
if args.dataset=='SMAL':
  template_fp = osp.join('template', 'smal_0.ply')
elif args.dataset=='DFaust':
  template_fp = osp.join('template', 'smpl_male_template.ply')
elif args.dataset=='Bone':
  template_fp = osp.join(args.data_dir, 'template.obj')
else:
    print('invalid dataset!')
    exit(-1)

meshdata = MeshData(args.data_fp,
                    template_fp,
                    dataset=args.dataset, 
                    pca_n_comp=args.latent_channels,
                    vert_pca=args.use_vert_pca)

train_loader = DataLoader(meshdata.train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True)
test_loader = DataLoader(meshdata.test_dataset, batch_size=args.batch_size, shuffle=True)

# generate/load transform matrices
transform_fp = osp.join(args.data_fp, 'transform.pkl')
if not osp.exists(transform_fp):
    print('Generating transform matrices...')
    mesh = Mesh(filename=template_fp)
    # ds_factors = [4, 4, 4, 4]
    #ds_factors = [2, 2, 2, 2]
    ds_factors = [1, 1, 1, 1]
    _, A, D, U, F = mesh_sampling.generate_transform_matrices(mesh, ds_factors)
    tmp = {'face': F, 'adj': A, 'down_transform': D, 'up_transform': U}

    with open(transform_fp, 'wb') as fp:
        pickle.dump(tmp, fp)
    print('Done!')
    print('Transform matrices are saved in \'{}\''.format(transform_fp))
else:
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

edge_index_list = [utils.to_edge_index(adj).to(device) for adj in tmp['adj']]
down_transform_list = [
    utils.to_sparse(down_transform).to(device)
    for down_transform in tmp['down_transform']
]
up_transform_list = [
    utils.to_sparse(up_transform).to(device)
    for up_transform in tmp['up_transform']
]

model = AE(args.in_channels,
           args.out_channels,
           args.latent_channels,
           edge_index_list,
           down_transform_list,
           up_transform_list,
           K=args.K)

model.to(device)
#print(model)

if args.use_vert_pca:
    pca_init = torch.from_numpy(meshdata.train_pca_sv)
    lat_vecs = torch.nn.Embedding.from_pretrained(pca_init, freeze=False)
elif args.use_pose_init:
    pose_init = torch.from_numpy(np.array(meshdata.train_dataset.data.pose, np.float32))
    pose_init = pose_init.reshape(meshdata.num_train_graph, -1)
    lat_vecs = torch.nn.Embedding.from_pretrained(pose_init, freeze=False)
else:
    train_num_scenes = len(meshdata.train_dataset)
    lat_vecs = torch.nn.Embedding(train_num_scenes, args.latent_channels,)
    torch.nn.init.normal_(lat_vecs.weight.data,0.0,0.2)

lat_vecs = lat_vecs.to(device)
print("train lat_vecs", lat_vecs)

test_num_scenes = len(meshdata.test_dataset)
test_lat_vecs = torch.nn.Embedding(test_num_scenes, args.latent_channels,)
torch.nn.init.normal_(test_lat_vecs.weight.data, 0.0, 0.2) 
test_lat_vecs = test_lat_vecs.to(device)
print("test lat_vecs", test_lat_vecs)

optimizer_all = torch.optim.Adam(
    [
        {
            "params": model.parameters(),
            "lr": args.lr,
            "weight_decay": args.weight_decay
        },
        {
            "params": lat_vecs.parameters(),
            "lr": args.lr,
            "weight_decay": args.weight_decay
        },
    ]
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer_all,
                                            args.decay_step,
                                            gamma=args.lr_decay)
if args.mode=='train':
    train_eval.run(model, train_loader, test_loader, lat_vecs, args.epochs, optimizer_all,\
                   scheduler, writer, device, results_dir_train, meshdata.mean.numpy(), meshdata.std.numpy(),meshdata.template_face,\
                   arap_weight=args.arap_weight, use_arap_epoch=args.use_arap_epoch, arap_eig_k=args.arap_eig_k, \
                           continue_train=args.continue_train,)

elif args.mode=='test':
    optimizer_test = torch.optim.Adam(test_lat_vecs.parameters(),
                                 lr=args.test_lr,
                                 weight_decay=args.weight_decay)
    scheduler_test = torch.optim.lr_scheduler.StepLR(optimizer_test,
                                            args.test_decay_step,
                                            gamma=args.lr_decay)
    train_eval.test_reconstruct(model, test_loader, test_lat_vecs, args.test_epochs, optimizer_test, scheduler_test, writer,
        device, results_dir_test, meshdata.mean.numpy(), meshdata.std.numpy(), meshdata.template_face)

elif args.mode=='interpolate':
    train_eval.global_interpolate(model, lat_vecs, optimizer_all, scheduler, 
       writer, device, args.results_dir, meshdata.mean.numpy(), meshdata.std.numpy(), meshdata.template_face, args.inter_num) 
elif args.mode=='extraplate':
    optimizer_test = torch.optim.Adam(test_lat_vecs.parameters(),
                                 lr=args.test_lr,
                                 weight_decay=args.weight_decay)
    scheduler_test = torch.optim.lr_scheduler.StepLR(optimizer_test,
                                            args.test_decay_step,
                                            gamma=args.lr_decay)
    train_eval.extrapolation(model, test_lat_vecs, optimizer_test, scheduler_test,
       writer, device, args.results_dir, meshdata.mean.numpy(), meshdata.std.numpy(), meshdata.template_face, args.extra_num, args.extra_thres)

