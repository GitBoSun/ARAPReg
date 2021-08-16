import time
import os, sys
import torch
import torch.nn.functional as F
from psbody.mesh import Mesh
import numpy as np
from models import ARAP
import scipy.io as sio
from tqdm import tqdm

def run(model, 
        train_loader, lat_vecs, optimizer, scheduler, 
        test_loader, test_lat_vecs, optimizer_test, scheduler_test,
        epochs, writer, device, results_dir, data_mean, data_std,
        template_face, arap_weight=0.0, use_arap_epoch=800, arap_eig_k=60, 
        continue_train=False, checkpoint=None, test_checkpoint=None):
    
    start_epoch = 0
    if continue_train:
        start_epoch = writer.load_checkpoint(
            model, lat_vecs, optimizer, scheduler, checkpoint=checkpoint)
    
    if (optimizer_test is not None) and continue_train:
        test_epoch = writer.load_checkpoint(
            model, test_lat_vecs, optimizer_test, scheduler_test,
            checkpoint=test_checkpoint, test=True)

    for epoch in range(start_epoch+1, epochs):
        t = time.time()
        use_arap = epoch > use_arap_epoch

        train_loss, l1_loss, arap_loss, l2_error = \
            train(
                model, epoch, optimizer, train_loader, lat_vecs,
                device, results_dir, data_mean, data_std, template_face,
                arap_weight=arap_weight, arap_eig_k=arap_eig_k,
                use_arap=use_arap, lr=scheduler.get_lr()[0],
            )

        if optimizer_test is not None:
            for k in range(3):
                test_loss, test_l1_loss, test_arap_loss, test_l2_error = \
                    train(
                        model, epoch*3+k, optimizer_test, test_loader, test_lat_vecs,
                        device, results_dir, data_mean, data_std, template_face,
                        arap_weight=0.0, arap_eig_k=arap_eig_k, use_arap=False,
                        exp_name='reconstruct', lr=scheduler_test.get_lr()[0],
                    )
                test_info = {
                    'test_current_epoch': epoch*3+k,
                    'test_epochs': epochs*3,
                    'test_loss': test_loss,
                    'l1_loss': test_l1_loss,
                    'arap_loss': test_arap_loss,
                    'mse_error': test_l2_error,
                    't_duration': 0.0,
                    'lr': scheduler_test.get_lr()[0]
                }
                scheduler_test.step()
                writer.print_info_test(test_info)
            if epoch % 10 == 0:
                writer.save_checkpoint(
                    model, test_lat_vecs, optimizer_test, scheduler_test,
                    epoch*3, test=True
                )

        t_duration = time.time() - t
        scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'l1_loss': l1_loss,
            'arap_loss': arap_loss,
            'mse_error': l2_error,
            't_duration': t_duration,
            'lr': scheduler.get_last_lr()[0]
        }

        writer.print_info(info)
        if epoch % 10 == 0:
            writer.save_checkpoint(model, lat_vecs, optimizer, scheduler, epoch)


def train(model, epoch, optimizer, loader, lat_vecs, device,
          results_dir,data_mean, data_std, template_face, 
          arap_weight=0.0, arap_eig_k=30, use_arap=False, dump=False,
          exp_name='train', lr=0.0):
    
    model.train()
    total_loss = 0
    total_l1_loss = 0
    total_arap_loss = 0
    l2_errors = []
    
    data_mean_gpu = torch.as_tensor(data_mean, dtype=torch.float).to(device)
    data_std_gpu = torch.as_tensor(data_std, dtype=torch.float).to(device)

    model.train()
    pbar = tqdm(total=len(loader),
                desc=f'{exp_name} {epoch}')

    # ARAP
    arap = ARAP(template_face, template_face.max()+1).to(device)
    mse = [0, 0]

    for b, data in enumerate(loader):
        optimizer.zero_grad()
        x = data.x.to(device)
        #x[:] = loader.dataset[4044].x
        ids = data.data_id.to(device)
        #ids[:] = 4044
        batch_vecs = lat_vecs(ids.view(-1))
        out = model(batch_vecs)
                
        pred_shape = out * data_std_gpu + data_mean_gpu
        gt_shape = x * data_std_gpu + data_mean_gpu
        
        # l1_loss = F.l1_loss(pred_shape, gt_shape, reduction='mean') 
        #l1_loss = F.l1_loss(x, out, reduction='mean')
        l1_loss = F.l1_loss(pred_shape, gt_shape, reduction='mean')

        tmp_error = torch.sqrt(torch.sum((pred_shape - gt_shape)**2,dim=2)).detach().cpu()
        mse[1] += tmp_error.view(-1).shape[0]
        mse[0] += tmp_error.sum().item()
        l2_errors.append(tmp_error)
        
        loss = torch.zeros(1).to(device) 
        loss += l1_loss

        if use_arap and arap_weight>0:
            
            jacob = get_jacobian_rand(
                        pred_shape,  batch_vecs, data_mean_gpu, data_std_gpu,
                        model, device,
                        epsilon=1e-1)
            arap_energy = arap(pred_shape, jacob, k=arap_eig_k) / jacob.shape[-1]
            total_arap_loss += arap_energy.item()
            loss += arap_weight*arap_energy

        loss.backward()
        total_loss += loss.item()
        total_l1_loss += l1_loss.item()
        pbar.set_postfix({
            'loss': total_loss / (b+1.0),
            'arap': total_arap_loss / (b+1.0),
            'w_arap': arap_weight,
            'MSE': mse[0] / (mse[1]+1e-6) * 1000,
            'lr': lr,
            })
        pbar.update(1)
        optimizer.step()
    new_errors = torch.cat(l2_errors, dim=0) 
    mean_error = new_errors.view((-1, )).mean()

    # visualize
    #if epoch%20==0 and dump==False:
    #    gt_meshes = x.detach().cpu().numpy()
    #    pred_meshes = out.detach().cpu().numpy()
    #    for b_i in range(min(2, gt_meshes.shape[0])):
    #        pred_v = pred_meshes[b_i].reshape((-1, 3))*data_std + data_mean
    #        gt_v = gt_meshes[b_i].reshape((-1, 3))*data_std + data_mean
    #        mesh = Mesh(v=pred_v, f=template_face)
    #        mesh.write_ply(os.path.join(results_dir, '%d_%d'%(epoch, b_i)+'_pred.ply'))
    #        mesh = Mesh(v=gt_v, f=template_face)
    #        mesh.write_ply(os.path.join(results_dir, '%d_%d'%(epoch, b_i)+'_gt.ply'))
    #if dump==True and epoch%200==1:
    #    model.eval()
    #    with torch.no_grad():
    #        for data in loader:
    #            x = data.x.to(device)
    #            ids = data.data_id.to(device)
    #            batch_vecs = lat_vecs(ids.view(-1))
    #            out = model(batch_vecs)
    #            pred_shape = out * data_std_gpu + data_mean_gpu
    #            gt_shape = x * data_std_gpu + data_mean_gpu
    #            gt_meshes = gt_shape.detach().cpu().numpy()
    #            pred_meshes = pred_shape.detach().cpu().numpy()
    #            ids_np = data.data_id.cpu().numpy()

    #            for b_i in range(gt_meshes.shape[0]):
    #                pred_v = pred_meshes[b_i].reshape((-1, 3))
    #                gt_v = gt_meshes[b_i].reshape((-1 , 3))
    #                mesh = Mesh(v=pred_v, f=template_face)
    #                mesh.write_ply(os.path.join(results_dir, '%d'%(ids_np[b_i])+'_pred.ply'))
    #                mesh = Mesh(v=gt_v, f=template_face)
    #                mesh.write_ply(os.path.join(results_dir, '%d'%(ids_np[b_i])+'_gt.ply'))

    return total_loss / len(loader), total_l1_loss / len(loader), total_arap_loss / len(loader), mean_error

def get_jacobian(out, z, data_mean_gpu, data_std_gpu, model, device, epsilon=[1e-3]):
    nb, nz = z.size()
    _, n_vert, nc = out.size()
    jacobian = torch.zeros((nb, n_vert*nc, nz)).to(device)

    for i in range(nz):
        dz = torch.zeros(z.size()).to(device)
        dz[:, i] = epsilon
        z_new = z + dz
        out_new = model(z_new)
        dout = (out_new - out).view(nb, -1)
        jacobian[:, :, i] = dout/epsilon

    data_std_gpu = data_std_gpu.reshape((1, n_vert*nc, 1))
    jacobian = jacobian*data_std_gpu
    return jacobian

def get_jacobian_k4(out, z, data_mean_gpu, data_std_gpu, model, device, epsilon=1e-3):
    nb, nz = z.size()
    _, n_vert, nc = out.size()
    jacobian = torch.zeros((nb, n_vert*nc, nz)).to(device)
    k = 4
    for i in range(nz//k):
        batched_z_new = []
        for j in range(k):
            dz = torch.zeros(z.size()).to(device)
            dz[:, i*k+j] = epsilon
            z_new = z + dz
            batched_z_new.append(z_new)
        batched_z_new = torch.stack(batched_z_new)
        out_new = model(batched_z_new)

        dout = (out_new.reshape(k, nb, n_vert, nc) - out.unsqueeze(0)).view(k, nb*n_vert*nc).transpose(0, 1).reshape(nb,n_vert*nc,k)

        jacobian[:, :, i*k:(i+1)*k] = dout/epsilon

    data_std_gpu = data_std_gpu.reshape((1, n_vert*nc, 1))
    jacobian = jacobian*data_std_gpu
    return jacobian

def get_jacobian_rand(cur_shape, z, data_mean_gpu, data_std_gpu, model, device, epsilon=[1e-3]):
    nb, nz = z.size()
    _, n_vert, nc = cur_shape.size()
    if nz >= 10:
      rand_idx = np.random.permutation(nz)[:10]
      nz = 10
    else:
      rand_idx = np.arange(nz)
    
    jacobian = torch.zeros((nb, n_vert*nc, nz)).to(device)
    for i, idx in enumerate(rand_idx):
        dz = torch.zeros(z.size()).to(device)
        dz[:, idx] = epsilon
        z_new = z + dz
        out_new = model(z_new)
        shape_new = out_new * data_std_gpu + data_mean_gpu
        dout = (shape_new - cur_shape).view(nb, -1)
        jacobian[:, :, i] = dout/epsilon
    return jacobian


def test_reconstruct(
    model, test_loader, test_lat_vecs, epochs, test_optimizer, scheduler,
    writer, device, results_dir, data_mean, data_std, template_face,
    checkpoint=None, test_checkpoint=None):
    # load model
    start_epoch = writer.load_checkpoint(model, checkpoint=checkpoint)
    if test_checkpoint is not None:
        start_epoch = writer.load_checkpoint(
            model, test_lat_vecs, test_optimizer, scheduler,
            checkpoint=test_checkpoint, test=True)

    for epoch in range(1, epochs + 1):
        t = time.time()

        test_loss, l1_loss, arap_loss, l2_error = \
            train(model, epoch, test_optimizer, test_loader, 
                test_lat_vecs, device, results_dir, data_mean,
                data_std, template_face, arap_weight=0.0, use_arap=False,
                dump=True, exp_name='reconstruct', lr=scheduler.get_lr()[0],
            )
        
        t_duration = time.time() - t
        scheduler.step()
        info = {
            'test_current_epoch': epoch,
            'test_epochs': epochs,
            'test_loss': test_loss,
            'l1_loss': l1_loss,
            'arap_loss':arap_loss,
            'mse_error':l2_error,
            't_duration': t_duration,
            'lr': scheduler.get_lr()[0]
        }
        if epoch % 200 == 1:
            writer.save_checkpoint(model, test_lat_vecs, test_optimizer,
                scheduler, epoch, test=True)
        writer.print_info_test(info)
    writer.save_checkpoint(
        model, test_lat_vecs, test_optimizer,
        scheduler, epoch, test=True)

def global_interpolate(
        model, lat_vecs, optimizer, scheduler, writer,
        device, results_dir, data_mean, data_std,
        template_face, interpolate_num):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    import numpy as np

    load_epoch = writer.load_checkpoint(model, lat_vecs, optimizer, scheduler,)
    
    lat_vecs_np = lat_vecs.weight.data.detach().cpu().numpy()
    results_dir = os.path.join(results_dir, "interpolate/epoch%d"%(load_epoch))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    min_path = np.random.randint(0,  min_tree.shape[0], size=50)
    print('min_path', min_path)

    for p in range(len(min_path)-1):
        start_vec = lat_vecs.weight.data[min_path[p]]
        end_vec = lat_vecs.weight.data[min_path[p+1]]

        for i in range(interpolate_num+1):
            vec = start_vec + i*(end_vec - start_vec)/interpolate_num
            out =  model(vec)
            out_numpy = out.detach().cpu().numpy()
            out_v = out_numpy.reshape((-1, 3))*data_std + data_mean

            mesh = Mesh(v=out_v, f=template_face)
            mesh.write_obj(os.path.join(results_dir, '%d_%d_%d-%d'%(p, i, min_path[p], min_path[p+1])+'.obj'))


def extrapolation(model, lat_vecs, optimizer, scheduler, writer, device, results_dir, data_mean,
        data_std, template_face, extra_num=5, extra_thres=0.2):

    load_epoch = writer.load_checkpoint(model, lat_vecs, optimizer, scheduler,test=True)

    lat_vecs_np = lat_vecs.weight.data.detach().cpu().numpy()
    print('lat_vecs_np', lat_vecs_np)
    results_dir = os.path.join(results_dir, "extrapolation/epoch%d"%(load_epoch))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    z_dict = {}
    num_train_scenes = lat_vecs_np.shape[0]

    z_path = np.random.randint(0,  num_train_scenes, size=30)
    print(z_path)
    for p in range(len(z_path)-1):
        or_vec = lat_vecs.weight.data[z_path[p]]
        extra_z = torch.zeros((extra_num+1, lat_vecs_np.shape[1])).to(device)

        for i in range(extra_num):
            vec = or_vec + extra_thres*(torch.rand(lat_vecs_np.shape[1]).to(device)-0.5)
            extra_z[i+1] = vec
        extra_z[0] = or_vec
        out =  model(extra_z)
        out_numpy = out.detach().cpu().numpy()
        for i in range(extra_num):
            out_v = out_numpy[i].reshape((-1, 3))*data_std + data_mean
            mesh = Mesh(v=out_v, f=template_face)
            mesh.write_ply(os.path.join(results_dir, '%d_%d'%(z_path[p],i)+'.ply'))
            z_dict['%d_%d'%( z_path[p], i)] = extra_z[i].detach().cpu().numpy()
    np.save(os.path.join(results_dir, "name_latentz.npy"), z_dict)


