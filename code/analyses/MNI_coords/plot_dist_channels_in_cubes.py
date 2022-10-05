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

side = 5

x_min, x_max = df['MNI_x'].min(), df['MNI_x'].max()
y_min, y_max = df['MNI_z'].min(), df['MNI_y'].max()
z_min, z_max = df['MNI_z'].min(), df['MNI_y'].max()

n_x, n_y, n_z = len(np.arange(x_min, x_max+side, side/2)), len(np.arange(y_min, y_max+side, side/2)), len(np.arange(z_min, z_max+side, side/2))

cnt = 0
n_channels = []
for x in np.arange(x_min, x_max+side, side/2):
    for y in np.arange(y_min, y_max+side, side/2):
        for z in np.arange(z_min, z_max+side, side/2):
            cnt += 1
            if cnt % 1000 == 1: print(f'{cnt}/{n_x * n_y * n_z}')
            df_cube = UtilsCoords.pick_channels_by_cube(df, (x, y, z),
                                                        side=side,
                                                        isMacro=True,
                                                        isMicro=False,
                                                        isStim=False)
            n_channels.append(df_cube.shape[0])
            
n_channels = np.asarray(n_channels)
plt.hist(n_channels[n_channels>0], bins=40)

print(f'non-empty voxels : {len([i for i in n_channels if i>0])}/{len(n_channels)}')
print(f'multivariate voxels : {len([i for i in n_channels if i>1])}/{len(n_channels)}')
