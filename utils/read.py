import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import openmesh as om

def read_mesh(path, data_id, pose=None, return_face=False):
    mesh = om.read_trimesh(path)
    points = mesh.points()
    face = torch.from_numpy(mesh.face_vertex_indices()).T.type(torch.long)

    x = torch.tensor(points.astype('float32'))
    edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    edge_index = to_undirected(edge_index)
    if return_face==True and pose is not None:
        return Data(x=x, edge_index=edge_index,face=face, data_id=data_id, pose=pose)
    if pose is not None:
        return Data(x=x, edge_index=edge_index, data_id=data_id, pose=pose)
    if return_face==True:
        return Data(x=x, edge_index=edge_index,face=face, data_id=data_id)
    return Data(x=x, edge_index=edge_index, data_id=data_id)

