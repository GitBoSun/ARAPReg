import torch
from torch_geometric.utils import degree, get_laplacian
import torch_sparse as ts
import numpy as np
import sys

def get_laplacian_kron3x3(edge_index, edge_weights, N):
  edge_index, edge_weight = get_laplacian(edge_index, edge_weights, num_nodes=N)
  edge_weight *= 2
  e0, e1 = edge_index
  i0 = [e0*3, e0*3+1, e0*3+2]
  i1 = [e1*3, e1*3+1, e1*3+2]
  vals = [edge_weight, edge_weight, edge_weight]
  i0 = torch.cat(i0, 0)
  i1 = torch.cat(i1, 0)
  vals = torch.cat(vals, 0)
  indices, vals = ts.coalesce([i0, i1], vals, N*3, N*3)
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
    edge_weight = torch.ones_like(e0)
    
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
     
      e = torch.linalg.eigvalsh(JTLJ-JTBCBTJ).clip(0)

      e = e ** 0.5

      trace = e.sum()
      
      trace_.append(trace)
      
    trace_ = torch.stack(trace_, )
    return trace_.mean()
