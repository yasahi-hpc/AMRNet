import xarray as xr
import torch
import numpy as np
import pathlib

def save_as_netcdf(sdf, real_flows, pred_flows, indices, epoch, level, name, data_dir):
    # Convert to numpy array
    sdf = sdf.numpy()
    real_flows = real_flows.numpy()
    pred_flows = pred_flows.numpy()
    
    sub_dir = pathlib.Path(f'{data_dir}/{name}')
    if not sub_dir.exists():
        sub_dir.mkdir(parents=True)
     
    for sdf_, real_flows_, pred_flows_, idx in zip(sdf, real_flows, pred_flows, indices):
        filename = sub_dir / f'{name}{idx:06}_Lv{level}_epoch{epoch:04}.h5'
             
        # Check the dimension first
        shape = sdf_.shape
        dim = len(shape)

        if dim == 3:
            # Unpatched data (#channel, #h, #w)
            coords_list = ['y', 'x']
            data_vars = {}
            data_vars['SDF']   = (coords_list, sdf_[0])
            data_vars['u']     = (coords_list, real_flows_[0])
            data_vars['v']     = (coords_list, real_flows_[1])
            data_vars['u_hat'] = (coords_list, pred_flows_[0])
            data_vars['v_hat'] = (coords_list, pred_flows_[1])
            
            coords = {}
            _, ny, nx = shape
            coords['y'], coords['x'] = np.arange(ny), np.arange(nx)
            
            xr.Dataset(data_vars = data_vars, coords=coords).to_netcdf(filename, engine='netcdf4')
            
        else:
            # Patched data (#patch_h, #patch_x, #channel, #h, #w)
            data_vars = {}
            coords_list = ['patch_y', 'patch_x', 'y', 'x']
            data_vars = {}
            data_vars['SDF']   = (coords_list, sdf_[:,:,0])
            data_vars['u']     = (coords_list, real_flows_[:,:,0])
            data_vars['v']     = (coords_list, real_flows_[:,:,1])
            data_vars['u_hat'] = (coords_list, pred_flows_[:,:,0])
            data_vars['v_hat'] = (coords_list, pred_flows_[:,:,1])
            
            coords = {}
            py, px, _, ny, nx = shape
            coords['y'], coords['x'] = np.arange(ny), np.arange(nx)
            coords['patch_y'], coords['patch_x'] = np.arange(py), np.arange(px)
            
            xr.Dataset(data_vars = data_vars, coords=coords).to_netcdf(filename, engine='netcdf4')
