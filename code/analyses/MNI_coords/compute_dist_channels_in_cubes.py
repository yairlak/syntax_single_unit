#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 22:24:17 2022

@author: yair
"""
import UtilsCoords
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

path2data = '../../../Data/UCLA/MNI_coords/'

fn_coords = 'electrode_locations.csv'
df = pd.read_csv(os.path.join(path2data, fn_coords))

for side in [64, 32, 24, 16, 8, 4]:
    fn = f'../../../Output/MNI_coords/n_multivariate_voxels_with_side_{side}.csv'
    
    if os.path.isfile(fn):
        print(f'Already exists: {fn}, skipping')
        continue
    #stride = side/2
    stride = side
    print(f'side = {side}, stride = {stride}')

    x_min, x_max = df['MNI_x'].min(), df['MNI_x'].max()
    y_min, y_max = df['MNI_z'].min(), df['MNI_y'].max()
    z_min, z_max = df['MNI_z'].min(), df['MNI_y'].max()

    n_x, n_y, n_z = len(np.arange(x_min, x_max+side, stride)), len(np.arange(y_min, y_max+side, stride)), len(np.arange(z_min, z_max+side, stride))

    cnt = 0
    n_channels = []
    for x in np.arange(x_min, x_max+side, stride):
        for y in np.arange(y_min, y_max+side, stride):
            for z in np.arange(z_min, z_max+side, stride):
                cnt += 1
                if cnt % 1000 == 1: print(f'{cnt}/{n_x * n_y * n_z}')
                df_cube = UtilsCoords.pick_channels_by_cube(df, (x, y, z),
                                                            side=side,
                                                            isMacro=True,
                                                            isMicro=False,
                                                            isStim=False)
                n_channels.append(df_cube.shape[0])
                
    n_channels = np.asarray(n_channels)
    
    # MULTIVARIATE VOXELS 
    non_empty_voxels = len([i for i in n_channels if i>0])
    multivariate_voxels = len([i for i in n_channels if i>1])
    print(f'non-empty voxels : {non_empty_voxels}')
    print(f'multivariate voxels : {multivariate_voxels}')

    # SAVE
    with open(fn, 'w') as f:
        f.write(f'non_empty_voxels, multivariate_voxels, total\n')
        f.write(f'{non_empty_voxels}, {multivariate_voxels}, {n_x * n_y * n_z}\n')
    print(f'csv saved to: {fn}')
    
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist(n_channels[n_channels>0], bins=40)
    ax.set_title(f'Side of cube = {side}mm', fontsize=16)
    ax.set_xlabel('Number of channels in cube')
    ax.set_ylabel('Number of cubes')
    fn_fig = f'../../../Figures/MNI_coords/n_multivariate_voxels_with_side_{side}.png'
    fig.savefig(fn_fig)
    print(f'Figure saved to: {fn_fig}')
