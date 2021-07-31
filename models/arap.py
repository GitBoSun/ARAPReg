import torch
from torch_geometric.utils import degree, get_laplacian
from torch_scatter import scatter_add
import torch_sparse as ts
import numpy as np
import scipy.io as sio
import sys

def get_laplacian_kron3x3(edge_index, edge_weights, N):
  edge_index, edge_weight = get_laplacian(edge_index, edge_weights, num_nodes=N)
  edge_weight *= 2
  e0, e1 = edge_index
  i0 = [e0*3, e0*3+1, e0*3+2]
  i1 = [e1*3, e1*3+1, e1*3+2]
  vals = [edge_weight, edge_weight, edge_weight]
  #indices.append((e0*3+1)*N*3+e1*3+1)
  #indices.append((e0*3+2)*N*3+e1*3+2)
  #vals.append(edge_weight)
  #vals.append(edge_weight)
  #vals.append(edge_weight)
  i0 = torch.cat(i0, 0)
  i1 = torch.cat(i1, 0)
  vals = torch.cat(vals, 0)
  indices, vals = ts.coalesce([i0, i1], vals, N*3, N*3)
  #L = scatter_add(vals, indices, dim_size=(N*3*N*3)).view(N*3, N*3)
  return indices, vals

class ARAP(torch.nn.Module):
  def __init__(self, template_face, num_points):
    super(ARAP, self).__init__()
    N = num_points
    self.template_face = template_face
    adj = np.zeros((num_points, num_points))
    adj[template_face[:, 0], template_face[:, 1]] = 1
    adj[template_face[:, 1], template_face[:, 2]] = 1
    adj[template_face[:, 0], template_face[:, 2]] = 1
    adj = adj + adj.T
    edge_index = torch.as_tensor(np.stack(np.where(adj > 0), 0),
                                 dtype=torch.long)
    e0, e1 = edge_index
    deg = degree(e0, N)
    edge_weight = torch.ones_like(e0) #1.0/(deg[e0] + deg[e1])
    
    L_indices, L_vals = get_laplacian_kron3x3(edge_index, edge_weight, N) 
    self.register_buffer('L_indices', L_indices)
    self.register_buffer('L_vals', L_vals)
    self.register_buffer('edge_weight', edge_weight)
    self.register_buffer('edge_index', edge_index)

  def forward(self, x, J, k=0):
    """
      x: [B, N, 3] point locations.
      J: [B, N*3, D] Jacobian of generator.
      J_eigvals: [B, D]
    """
    num_batches, N = x.shape[:2]
    e0, e1 = self.edge_index
    #edge_vecs = (x[:, e0, :] - x[:, e1, :]).detach()
    edge_vecs = x[:, e0, :] - x[:, e1, :]
    trace_ = []
    
    for i in range(num_batches):
      LJ = ts.spmm(self.L_indices, self.L_vals, N*3, N*3, J[i])
      JTLJ = J[i].T.matmul(LJ)
      
      B0, B1, B_vals = [], [], []
      B0.append(e0*3  ); B1.append(e1*3+1); B_vals.append(-edge_vecs[i, :, 2]*self.edge_weight)
      B0.append(e0*3  ); B1.append(e1*3+2); B_vals.append( edge_vecs[i, :, 1]*self.edge_weight)
      B0.append(e0*3+1); B1.append(e1*3+0); B_vals.append( edge_vecs[i, :, 2]*self.edge_weight)
      B0.append(e0*3+1); B1.append(e1*3+2); B_vals.append(-edge_vecs[i, :, 0]*self.edge_weight)
      B0.append(e0*3+2); B1.append(e1*3+0); B_vals.append(-edge_vecs[i, :, 1]*self.edge_weight)
      B0.append(e0*3+2); B1.append(e1*3+1); B_vals.append( edge_vecs[i, :, 0]*self.edge_weight)

      B0.append(e0*3  ); B1.append(e0*3+1); B_vals.append(-edge_vecs[i, :, 2]*self.edge_weight)
      B0.append(e0*3  ); B1.append(e0*3+2); B_vals.append( edge_vecs[i, :, 1]*self.edge_weight)
      B0.append(e0*3+1); B1.append(e0*3+0); B_vals.append( edge_vecs[i, :, 2]*self.edge_weight)
      B0.append(e0*3+1); B1.append(e0*3+2); B_vals.append(-edge_vecs[i, :, 0]*self.edge_weight)
      B0.append(e0*3+2); B1.append(e0*3+0); B_vals.append(-edge_vecs[i, :, 1]*self.edge_weight)
      B0.append(e0*3+2); B1.append(e0*3+1); B_vals.append( edge_vecs[i, :, 0]*self.edge_weight)
      B0 = torch.cat(B0, 0)
      B1 = torch.cat(B1, 0)
      B_vals = torch.cat(B_vals, 0)
      B_indices, B_vals = ts.coalesce([B0, B1], B_vals, N*3, N*3)
      BT_indices, BT_vals = ts.transpose(B_indices, B_vals, N*3, N*3)
      
      C0, C1, C_vals = [], [], []
      edge_vecs_sq = (edge_vecs[i] * edge_vecs[i]).sum(-1)
      evi = edge_vecs[i]
      for di in range(3):
        for dj in range(3):
          C0.append(e0*3+di); C1.append(e0*3+dj); C_vals.append(-evi[:, di]*evi[:, dj]*self.edge_weight)
        C0.append(e0*3+di); C1.append(e0*3+di); C_vals.append(edge_vecs_sq*self.edge_weight)
      C0 = torch.cat(C0, 0)
      C1 = torch.cat(C1, 0)
      C_vals = torch.cat(C_vals, 0)
      C_indices, C_vals = ts.coalesce([C0, C1], C_vals, N*3, N*3)
      C_vals = C_vals.view(N, 3, 3).inverse().reshape(-1)
      BTJ = ts.spmm(BT_indices, BT_vals, N*3, N*3, J[i])
      CBTJ = ts.spmm(C_indices, C_vals, N*3, N*3, BTJ)
      JTBCBTJ = BTJ.T.mm(CBTJ)
      
      
      e,v = torch.eig(JTLJ-JTBCBTJ, eigenvectors=True)
      
      trace = e[k:,0].sum()
      #trace = torch.sqrt(e[:,0]).sum()
      # trace = (JTLJ-JTBCBTJ).trace()
      trace_.append(trace)
      
    trace_ = torch.stack(trace_, )
    return trace_.mean()
    
    #B0, B1, B_vals = [], [], []
    ## Part 1
    #B_indices.append((e0*3+0)*N*3+(e1*3+1))
    #B_vals.append( edge_vecs[:, :, 2]*self.edge_weight)
    #B_indices.append((e0*3+0)*N*3+(e1*3+2))
    #B_vals.append(-edge_vecs[:, :, 1]*self.edge_weight)
    #B_indices.append((e0*3+1)*N*3+(e1*3+2))
    #B_vals.append( edge_vecs[:, :, 0]*self.edge_weight)
    #B_indices.append((e0*3+1)*N*3+(e1*3+0))
    #B_vals.append(-edge_vecs[:, :, 2]*self.edge_weight)
    #B_indices.append((e0*3+2)*N*3+(e1*3+0))
    #B_vals.append( edge_vecs[:, :, 1]*self.edge_weight)
    #B_indices.append((e0*3+2)*N*3+(e1*3+1))
    #B_vals.append(-edge_vecs[:, :, 0]*self.edge_weight)
    ## Part 2
    #B_indices.append((e0*3+0)*N*3+(e0*3+1))
    #B_vals.append(-edge_vecs[:, :, 2]*self.edge_weight)
    #B_indices.append((e0*3+0)*N*3+(e0*3+2))
    #B_vals.append( edge_vecs[:, :, 1]*self.edge_weight)
    #B_indices.append((e0*3+1)*N*3+(e0*3+2))
    #B_vals.append(-edge_vecs[:, :, 0]*self.edge_weight)
    #B_indices.append((e0*3+1)*N*3+(e0*3+0))
    #B_vals.append( edge_vecs[:, :, 2]*self.edge_weight)
    #B_indices.append((e0*3+2)*N*3+(e0*3+0))
    #B_vals.append(-edge_vecs[:, :, 1]*self.edge_weight)
    #B_indices.append((e0*3+2)*N*3+(e0*3+1))
    #B_vals.append( edge_vecs[:, :, 0]*self.edge_weight)
    #B_indices = torch.cat(B_indices, 0)
    #B_vals = torch.cat(B_vals, 1)

    #C_indices, C_vals = [], []
    #edge_vecs_sq = (edge_vecs * edge_vecs).sum(-1)
    #for di in range(3):
    #  for dj in range(3):
    #    C_indices.append((e0*3+di)*3+dj)
    #    C_vals.append(-edge_vecs[:, :, di]*edge_vecs[:, :, dj]*self.edge_weight)
    #  C_indices.append((e0*3+di)*3+di)
    #  C_vals.append(edge_vecs_sq*self.edge_weight)
    #  
    #C_indices = torch.cat(C_indices, 0)
    #C_vals = torch.cat(C_vals, 1)
    #
    #Bmat, Cmat = [], []
    #i0, i1, i2 = np.meshgrid(np.arange(N), np.arange(3), np.arange(3),
    #                         indexing='ij')
    #x0 = i0*3+i1
    #y0 = i0*3+i2
    #index = torch.as_tensor(x0*N*3+y0, dtype=torch.long)
    #
    #for i in range(B):
    #  Bmat_i = scatter_add(B_vals[i], B_indices, dim_size=N*3*N*3)
    #  Bmat_i = -Bmat_i.view(N*3, N*3).T
    #  Bmat.append(Bmat_i)
    #  Cmat_i = scatter_add(C_vals[i], C_indices, dim_size=N*3*3).view(N, 9)
    #  #Cmat_i[e0.unsqueeze(-1), torch.as_tensor([0, 4, 8], dtype=torch.long)] += (edge_vecs[i] * edge_vecs[i]).sum(-1).unsqueeze(-1)
    #  #import ipdb; ipdb.set_trace()
    #  Cmat_i = Cmat_i.view(N, 3, 3).pinverse()
    #  Cmat_i = scatter_add(Cmat_i.view(-1), index.view(-1), dim_size=N*3*N*3)
    #  Cmat.append(Cmat_i.view(N*3, N*3))
    #Bmat = torch.stack(Bmat, 0)
    ##Bmat = Bmat + Bmat.transpose((1, 2))
    #Cmat = torch.stack(Cmat, 0)
    #Hessian = self.L.unsqueeze(0) - Bmat.matmul(Cmat).matmul(Bmat.transpose(-1, -2))
    #return Hessian

if __name__ == '__main__':
  import open3d as o3d
  import os.path as osp
  import numpy as np
  template_fp = osp.join('template', 'template.obj')
  mesh = o3d.io.read_triangle_mesh(template_fp)
  triangles = np.array(mesh.triangles).astype(np.int32)
  Npoints = np.array(mesh.vertices).shape[0]
  #adj = np.zeros((Npoints, Npoints))
  #adj[triangles[:, 0], triangles[:, 1]] = 1
  #adj[triangles[:, 1], triangles[:, 2]] = 1
  #adj[triangles[:, 0], triangles[:, 2]] = 1
  #adj = adj + adj.T
  #template_edge_index = torch.as_tensor(np.stack(np.where(adj > 0), 0),
  #                                      dtype=torch.long)
  arap = ARAP(triangles, Npoints)
  points = [] #torch.randn([16, Npoints, 3])
  jacob = []
  import scipy.io as sio
  for i in range(16):
    mat = sio.loadmat('debug{:02d}-1.mat'.format(0))
    points_i = mat['vertexPoss'].T
    points_i = torch.as_tensor(points_i, dtype=torch.float)
    points.append(points_i)
    jacob_i = mat['jacob']
    jacob_i = torch.as_tensor(jacob_i, dtype=torch.float)
    jacob.append(jacob_i)
    #  {'vertexPoss': points[i].cpu().numpy().T, 'faceVIds': triangles.T})
  points = torch.stack(points, 0)
  jacob = torch.stack(jacob, 0)
  H = arap(points, jacob)
