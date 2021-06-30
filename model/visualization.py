import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.style

def save_images(imgs, filename, title, n_cols=4, crange=(-1.0, 1.0), abs=False, show_patches=False):
    mpl.style.use('classic')
    fontsize = 28
    fontname = 'Times New Roman'
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('font',  family=fontname)
     
    title_font = {'fontname':fontname, 'size':fontsize, 'color':'black',
                  'verticalalignment':'bottom'}
    axis_font = {'fontname':fontname, 'size':fontsize}

    shape = imgs.shape
    dim = len(shape)

    min_val, max_val = crange
    n_samples = len(imgs)
    n_rows = n_samples // n_cols


    if type(imgs) != np.ndarray:
        if imgs.device == 'cpu':
            imgs = imgs.numpy()
        else:
            imgs = imgs.cpu().numpy()

    if abs:
        imgs = np.abs(imgs)

    if dim == 3:
        # Unpatched
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12,12), subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))
        for i, ax in np.ndenumerate(axes.ravel()):
            if abs:
                im = ax.imshow(imgs[i], cmap='jet', origin = 'lower', vmin = min_val, vmax=max_val)
            else:
                im = ax.imshow(imgs[i], cmap='seismic', origin = 'lower', vmin = min_val, vmax=max_val)
        fig.colorbar(im, ax=axes.ravel().tolist())
        fig.suptitle(title, **title_font, y=0.9)
                              
        fig.savefig(filename)
        plt.close('all')
    else:
        # Patched
        n_patch_y, n_patch_x, ny_in_patch, nx_in_patch = shape[1], shape[2], shape[3], shape[4]
        ny, nx = ny_in_patch * n_patch_y, nx_in_patch * n_patch_x
        reshaped_data = np.zeros((n_samples, ny, nx))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12,12), subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))

        slice_unpatch = [slice(None)]*3
        patched_lines = []
        for iy_in_patch in range(n_patch_y):
            for ix_in_patch in range(n_patch_x):
                x_start = ix_in_patch * nx_in_patch
                y_start = iy_in_patch * ny_in_patch
                slice_unpatch[1] = slice(y_start, y_start+ny_in_patch)
                slice_unpatch[2] = slice(x_start, x_start+nx_in_patch)
                
                reshaped_data[tuple(slice_unpatch)] = imgs[:,iy_in_patch,ix_in_patch,:,:]

                # Append vertical lines
                if ix_in_patch != 0:
                    vertical_line = ([x_start, x_start], [0, ny-1])
                    patched_lines.append(vertical_line)

            # Append horizontal lines
            if iy_in_patch != 0:
                horizontal_line = ([0, nx-1], [y_start, y_start])
                patched_lines.append(horizontal_line)

        for i, ax in np.ndenumerate(axes.ravel()):
            if abs:
                im = ax.imshow(reshaped_data[i], cmap='jet', origin = 'lower', vmin = min_val, vmax=max_val)
            else:
                im = ax.imshow(reshaped_data[i], cmap='seismic', origin = 'lower', vmin = min_val, vmax=max_val)

            if show_patches:
                for line in patched_lines:
                    x_range, y_range = line
                    ax.plot(x_range, y_range, '--k', lw=0.5)

            ax.set_xlim(0, nx)
            ax.set_ylim(0, ny)

        fig.colorbar(im, ax=axes.ravel().tolist())
        fig.suptitle(title, **title_font, y=0.9)
                              
        fig.savefig(filename)
        plt.close('all')

def save_flows(flows, name, img_dir, type_name, level, epoch, abs=False, show_patches=False):
    to_abs = True if type_name == 'error' else False
    crange = (-1.0, 1.0)
    batch_len = len(flows)
    n_cols = int(np.sqrt(batch_len+0.001))
    if to_abs:
        u_title = r"{} $\vert u \vert$ (Lv{}) epoch {:03}".format(type_name, level, epoch)
        v_title = r"{} $\vert v \vert$ (Lv{}) epoch {:03}".format(type_name, level, epoch)
        crange = (0.0, 1.0)
    else:
        u_title = r"{} $u$ (Lv{}) epoch {:03}".format(type_name, level, epoch)
        v_title = r"{} $v$ (Lv{}) epoch {:03}".format(type_name, level, epoch)

    shape = flows.shape
    dim = len(shape)

    if dim == 4:
        save_images(flows[:, 0, :, :],
                    f"{img_dir}/{name}_Lv{level}/{name}_{type_name}_Lv{level}_u_epoch_{epoch:03}.png", 
                    u_title, n_cols=n_cols, crange=crange, abs=to_abs, show_patches=show_patches)
        save_images(flows[:, 1, :, :],
                    f"{img_dir}/{name}_Lv{level}/{name}_{type_name}_Lv{level}_v_epoch_{epoch:03}.png", 
                    v_title, n_cols=n_cols, crange=crange, abs=to_abs, show_patches=show_patches)
    else:
        save_images(flows[:, :, :, 0],
                    f"{img_dir}/{name}_Lv{level}/{name}_{type_name}_Lv{level}_u_epoch_{epoch:03}.png", 
                    u_title, n_cols=n_cols, crange=crange, abs=to_abs, show_patches=show_patches)
        save_images(flows[:, :, :, 1],
                    f"{img_dir}/{name}_Lv{level}/{name}_{type_name}_Lv{level}_v_epoch_{epoch:03}.png", 
                    v_title, n_cols=n_cols, crange=crange, abs=to_abs, show_patches=show_patches)
