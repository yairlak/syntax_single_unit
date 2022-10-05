#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:19:32 2022

@author: yair
"""

import argparse
import os
import pandas as pd
import numpy as np

cluster = True
side = 5
smooth = 50
decimate = 50
comparison_name = 'dec_quest_len2'

stride = side/2


queue = 'Nspin_long'
walltime = '2:00:00'

# LOAD COORDINATES
path2coords = '../../Data/UCLA/MNI_coords/'
fn_coords = 'electrode_locations.csv'
df = pd.read_csv(os.path.join(path2coords, fn_coords))


x_min, x_max = df['MNI_x'].min(), df['MNI_x'].max()
y_min, y_max = df['MNI_z'].min(), df['MNI_y'].max()
z_min, z_max = df['MNI_z'].min(), df['MNI_y'].max()

n_x, n_y, n_z = len(np.arange(x_min, x_max+side, side/2)), len(np.arange(y_min, y_max+side, side/2)), len(np.arange(z_min, z_max+side, side/2))

cnt = 0
n_channels = []
for x in np.arange(x_min, x_max+side, stride):
    for y in np.arange(y_min, y_max+side, stride):
        for z in np.arange(z_min, z_max+side, stride):
            cnt += 1
            #if cnt % 1000 == 1: print(f'{cnt}/{n_x * n_y * n_z}')
            print(cnt)
            job_name = f'slight_{cnt}'
            output_log = f'slight_{cnt}.out'
            error_log = f'slight_{cnt}.err'
            
            # LAUNCH
            cmd = f'python decoding_searchlight.py --smooth {smooth} --decimate {decimate} --x {x} --y {y} --z {z} --side {side} --comparison-name {comparison_name}'
            if cluster:
                cmd = f"echo {cmd} | qsub -q {queue} -N {job_name} -l walltime={walltime} -o {output_log} -e {error_log}"
            
                
            os.system(cmd)
            