import torch
import dask.array as da
import xarray as xr
import numpy as np

class FlowDataset(torch.utils.data.Dataset):
    def __init__(self, files, transform = None, **kwargs):
        allowed_kwargs = {
                          'model_name',
                         }

        model_name = kwargs.get('model_name')
        if not model_name:
            raise ValueError('model_name must be specified')
        self.model_name = model_name
        
        self.files = files
        self.datanum = len(files)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        ds = xr.load_dataset(filename_or_obj=self.files[idx], engine='netcdf4')

        sdf_Lv0, u_Lv0, v_Lv0 = ds['SDF_lv0'], ds['u_lv0'], ds['v_lv0']
        sdf_Lv1, u_Lv1, v_Lv1 = ds['SDF_lv1'], ds['u_lv1'], ds['v_lv1']
        sdf_Lv2, u_Lv2, v_Lv2 = ds['SDF_lv2'], ds['u_lv2'], ds['v_lv2']

        if self.model_name in ['UNet', 'Pix2PixHD']:
            # Then, unpatch the data
            def trans(x, lv):
                x = x.transpose(f'patch_y_lv{lv}', f'y_lv{lv}', f'patch_x_lv{lv}', f'x_lv{lv}')
                x = x.stack(merged_y=[f'patch_y_lv{lv}', f'y_lv{lv}'], merged_x=[f'patch_x_lv{lv}', f'x_lv{lv}'])
                return x.values

            sdf_Lv0 = torch.tensor( np.expand_dims(trans(sdf_Lv0, 0), axis=0) ).float()
            sdf_Lv1 = torch.tensor( np.expand_dims(trans(sdf_Lv1, 1), axis=0) ).float()
            sdf_Lv2 = torch.tensor( np.expand_dims(trans(sdf_Lv2, 2), axis=0) ).float()

            flows_Lv0 = torch.tensor( np.stack([trans(u_Lv0, 0), trans(v_Lv0, 0)], axis=0) ).float()
            flows_Lv1 = torch.tensor( np.stack([trans(u_Lv1, 1), trans(v_Lv1, 1)], axis=0) ).float()
            flows_Lv2 = torch.tensor( np.stack([trans(u_Lv2, 2), trans(v_Lv2, 2)], axis=0) ).float()
        else:
            # In the patched layout
            sdf_Lv0 = torch.tensor( np.expand_dims(sdf_Lv0, axis=2) ).float()
            sdf_Lv1 = torch.tensor( np.expand_dims(sdf_Lv1, axis=2) ).float()
            sdf_Lv2 = torch.tensor( np.expand_dims(sdf_Lv2, axis=2) ).float()
            
            flows_Lv0 = torch.tensor( np.stack([u_Lv0, v_Lv0], axis=2) ).float()
            flows_Lv1 = torch.tensor( np.stack([u_Lv1, v_Lv1], axis=2) ).float()
            flows_Lv2 = torch.tensor( np.stack([u_Lv2, v_Lv2], axis=2) ).float()

        sdf  = (sdf_Lv0, sdf_Lv1, sdf_Lv2)
        flows = (flows_Lv0, flows_Lv1, flows_Lv2)

        return sdf, flows
