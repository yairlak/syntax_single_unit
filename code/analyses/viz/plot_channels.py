import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import ecog
import torch
import colorcet
from tqdm.notebook import tqdm, trange
import mne
import nibabel
# from ecog import read_data, releases, data_path

mne.set_log_level(0)


subjects_dir = '/volatile/freesurfer/subjects' # your freesurfer directory
patients_ = '479 482 499 502 504 505 510 513 515 530 538 539 540 541 543 544 545 549 551 552 553 554 556'.split()
# patients_ = '502'.split()
patients = ['patient_' + p for p in patients_]

path2data = os.path.join('..', '..', '..', 'Data', 'UCLA', 'MNI_coords')
fname = "/volatile/freesurfer/subjects/fsaverage/surf/"

# INIT FIGURE

# PLOT INFLATED BRAIN



def read_MNI_coord_from_xls(fn, data_type):
    assert data_type in ['micro', 'macro'] # only these two options allowed
    ismicro = (data_type == 'micro')
    
    df_coords = pd.read_excel(fn)
    df_coords['isMicro'] = df_coords['electrode'].str.contains('micro').values
    df_coords['isStim'] = df_coords['electrode'].str.contains('stim').values
    
    df_coords = df_coords[df_coords['isStim'] == False] # remove stim channels
    df_coords = df_coords[df_coords['isMicro'] == ismicro] # pick data_type (micro/macro)
    
    return df_coords
    # coords = [x for x in zip(df_coords['MNI_x'],
    #                          df_coords['MNI_y'],
    #                          df_coords['MNI_z'])]
    # return dict(zip(df_coords['electrode'], coords))


def get_distances(tris, channels, sigma=5):
    tris = np.nan_to_num(tris)
    # tris = torch.Tensor(tris).to("cuda")
    xyz = channels[["MNI_x", "MNI_y", "MNI_z"]].values.astype(np.float)
    xyz = np.nan_to_num(xyz)
    # xyz = torch.tensor(xyz).to("cuda")
    distances = torch.ones((len(xyz), len(tris)))#, device=xyz.device)
    for idx in range(len(xyz)):
        d = ((xyz[[idx]] - tris) ** 2).sum(1)
        if sigma > 0:
            d = torch.exp(-d / sigma ** 2)
        distances[idx, :] = torch.from_numpy(d)

    return distances.cpu().numpy()


def set_surface(channels, surf_path):

    pial = np.r_[
        mne.read_surface(surf_path + "lh.pial")[0],
        mne.read_surface(surf_path + "rh.pial")[0],
    ]

    infl = np.r_[
        mne.read_surface(surf_path + "lh.inflated_pre")[0],
        mne.read_surface(surf_path + "rh.inflated_pre")[0],
    ]

    sphere = np.r_[
        mne.read_surface(surf_path + "lh.sphere")[0],
        mne.read_surface(surf_path + "rh.sphere")[0],
    ]

    # compute distance
    coef = np.zeros((len(channels), len(pial)))
    for idx in np.array_split(range(len(channels)), 2):  # split for memory
        coef[idx] = get_distances(pial, channels.iloc[idx], sigma=0)
        # coef[idx] = get_distances(infl, channels.iloc[idx], sigma=0)

    # find closest vertex on pial
    idx = np.argmin(coef, 1)
    channels["closest_vert_idx"] = idx
    channels["distance_to_vert"] = np.array(
        [coef[j, i] for j, i in enumerate(idx)]
    )

    # project to vertex on inflated pre
    x, y, z = infl[channels.closest_vert_idx].T
    channels["surf_x"] = x
    channels["surf_y"] = y
    channels["surf_z"] = z

    x, y, z = sphere[channels.closest_vert_idx].T
    channels["sphere_x"] = x
    channels["sphere_y"] = y
    channels["sphere_z"] = z

    x, y, z = pial[channels.closest_vert_idx].T
    channels["pial_x"] = x
    channels["pial_y"] = y
    channels["pial_z"] = z

    return channels

for i_hemi, hemi in enumerate(['lh', 'rh']):
    fig, ax = plt.subplots(1, figsize=[16, 10])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "bottom", "right", "left"):
        ax.spines[s].set_visible(False)

    # tris, idx = mne.read_surface(fname + f"{hemi}.pial")
    tris_surf, _ = mne.read_surface(fname + f"{hemi}.inflated_pre")
    curv = nibabel.freesurfer.read_morph_data(fname + f"{hemi}.curv")
    
    # Normalize curvature to make gray cortex
    curv = (curv > 0).astype(float)
    curv = 1 - curv
    curv = (curv - 0.5) / 3 + 0.5
    curv /= 2.0
    curv = curv[:, np.newaxis] * [1, 1, 1, 1]
    curv[:, 3] = 1.0
    
    order = np.argsort(tris_surf[:, 0])
    
    # lr = (2 * i_hemi - 1)
    dots_r = ax.scatter(
        tris_surf[order, 1] + 120,
        tris_surf[order, 2],
        s=1,
        zorder=1,
        color=curv[order],
    )
    dots_l = ax.scatter(
        -tris_surf[order[::-1], 1] - 120,
        tris_surf[order[::-1], 2],
        s=1,
        zorder=1,
        color=curv[order[::-1]],
    )
    
    for data_type, color in zip(['micro', 'macro'], ['r', 'b']):
        dfs_coords = []
        for patient in patients:
            fn_coord = f'sub-{patient[8:11]}_space-MNI_electrodes.xlsx'        
            df_coords = read_MNI_coord_from_xls(os.path.join(path2data, fn_coord),
                                             data_type)
            dfs_coords.append(df_coords)
        channels = pd.concat(dfs_coords, axis=0)
    
        channels = set_surface(channels, fname)
        
        chs_l = channels.query("MNI_x<=0.7")[["surf_x", "surf_y", "surf_z"]]
        x, y, z = chs_l.values.T
        chs_l = ax.scatter(-y-120, z, s=3, color=color)
        
        
        chs_r = channels.query("MNI_x>=0.4")[["surf_x", "surf_y", "surf_z"]]
        x, y, z = chs_r.values.T
        chs_r = ax.scatter(-y+120, z, s=3, color=color)
    
    fn_fig = f'../../../Figures/viz_brain/electrode_locations_{hemi}_{"_".join(patients_)}.png'
    fig.savefig(fn_fig)
