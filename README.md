# AMRNet

_AMRNet_ is designed to predict steady flows from signed distance functions (SDF). 
Follwoing the landmarking work by [Guo et al](https://dl.acm.org/doi/10.1145/2939672.2939738), 
we extend the CNN prediction model to be applicable to the flow fields based on adaptive meshes. 
For this purpose, we employ the [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) based network to handle the data with multiple resolutions. 
The inputs of the network are multi-resolutional signed distance functions (SDFs). 
<p float="left">
  <img src="https://github.com/yasahi-hpc/AMRNet/blob/main/figs/SDF_Lv0_grid.png" width="300" />
  <img src="https://github.com/yasahi-hpc/AMRNet/blob/main/figs/SDF_Lv1_grid.png" width="300" /> 
  <img src="https://github.com/yasahi-hpc/AMRNet/blob/main/figs/SDF_Lv2_grid.png" width="300" />
</p>
We use the global (un-patched) data and local (patched) data to predict the multi-resolution flow fields.

# Usage

## Installation
This code relies on the following packages. As a deeplearing framework, we use [PyTorch](https://pytorch.org).
- Install  
[numpy](https://numpy.org), [PyTorch](https://pytorch.org), [xarray](http://xarray.pydata.org/en/stable/) and [netcdf4](https://github.com/Unidata/netcdf4-python)

- Clone this repo  
```git clone https://github.com/yasahi-hpc/AMRNet.git```


## Prepare dataset
The 2D flow dataset for AMR-Net has been computed by simulations using the lattice Boltzmann methods (LBMs). The inputs of simulations are signed distance functions (SDFs) and the outputs are 2D flow fields <img src="https://render.githubusercontent.com/render/math?math={u}"> and <img src="https://render.githubusercontent.com/render/math?math={v}">. Each data is stored in a hdf5 file in the following format.
```
<xarray.Dataset>
Dimensions:       (patch_x_lv0: 1, patch_x_lv1: 2, patch_x_lv2: 4, patch_y_lv0: 1, patch_y_lv1: 2, patch_y_lv2: 4, x_lv0: 256, x_lv1: 256, x_lv2: 256, y_lv0: 256, y_lv1: 256, y_lv2: 256)
Coordinates:
  * x_lv0         (x_lv0) int64 0 1 2 3 4 5 6 7 ... 249 250 251 252 253 254 255
  * y_lv0         (y_lv0) int64 0 1 2 3 4 5 6 7 ... 249 250 251 252 253 254 255
  * patch_x_lv0   (patch_x_lv0) int64 0
  * patch_y_lv0   (patch_y_lv0) int64 0
  * x_lv1         (x_lv1) int64 0 1 2 3 4 5 6 7 ... 249 250 251 252 253 254 255
  * y_lv1         (y_lv1) int64 0 1 2 3 4 5 6 7 ... 249 250 251 252 253 254 255
  * patch_x_lv1   (patch_x_lv1) int64 0 1
  * patch_y_lv1   (patch_y_lv1) int64 0 1
  * x_lv2         (x_lv2) int64 0 1 2 3 4 5 6 7 ... 249 250 251 252 253 254 255
  * y_lv2         (y_lv2) int64 0 1 2 3 4 5 6 7 ... 249 250 251 252 253 254 255
  * patch_x_lv2   (patch_x_lv2) int64 0 1 2 3
  * patch_y_lv2   (patch_y_lv2) int64 0 1 2 3
Data variables:
    SDF_lv0       (patch_y_lv0, patch_x_lv0, y_lv0, x_lv0) float32 ...
    x_starts_lv0  (patch_y_lv0, patch_x_lv0) float32 ...
    y_starts_lv0  (patch_y_lv0, patch_x_lv0) float32 ...
    SDF_lv1       (patch_y_lv1, patch_x_lv1, y_lv1, x_lv1) float32 ...
    x_starts_lv1  (patch_y_lv1, patch_x_lv1) float32 ...
    y_starts_lv1  (patch_y_lv1, patch_x_lv1) float32 ...
    SDF_lv2       (patch_y_lv2, patch_x_lv2, y_lv2, x_lv2) float32 ...
    x_starts_lv2  (patch_y_lv2, patch_x_lv2) float32 ...
    y_starts_lv2  (patch_y_lv2, patch_x_lv2) float32 ...
    u_lv0         (patch_y_lv0, patch_x_lv0, y_lv0, x_lv0) float32 ...
    u_lv1         (patch_y_lv1, patch_x_lv1, y_lv1, x_lv1) float32 ...
    u_lv2         (patch_y_lv2, patch_x_lv2, y_lv2, x_lv2) float32 ...
    v_lv0         (patch_y_lv0, patch_x_lv0, y_lv0, x_lv0) float32 ...
    v_lv1         (patch_y_lv1, patch_x_lv1, y_lv1, x_lv1) float32 ...
    v_lv2         (patch_y_lv2, patch_x_lv2, y_lv2, x_lv2) float32 ...
    m             int64 ...
```
The inputs are ```SDF_lv0 - SDF_lv2``` and outpus are ```u_lv0 - u_lv2, v_lv0 - v_lv2```. Each variable has the shape of ```(py, px, Ny, Nx)```, where ```px, py``` are the number of patches in x and y directions. ```Nx, Ny``` are the number of grid points in x and y directions inside a patch. 

The dataset can be downloaded from [Dataset]().


## Training

## Testing

# Summary
