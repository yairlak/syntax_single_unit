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
from MNI_coords import UtilsCoords

cluster = False
side = 8
smooth = 50
decimate = 50
comparison_name = 'dec_quest_len2'

stride = side/2


queue = 'Nspin_short'
walltime = '2:00:00'

# LOAD COORDINATES
path2code =  '/neurospin/unicog/protocols/intracranial/syntax_single_unit/code/analyses/'
path2code = '/home/yair/projects/syntax_single_unit/code/analyses'
logdir = 'logs'
script_name = 'decoding.py'

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
            # PICK CHANNELS IN CUBE
            df_cube = UtilsCoords.pick_channels_by_cube(df,
                                                        (x, y, z),
                                                        side=side,
                                                        isMacro=True,
                                                        isMicro=False,
                                                        isStim=False)
            # GET DATA AND DECODE
            if not df_cube.empty:
                cnt += 1
                #print(df_cube)
                
                patients = df_cube['patient'].astype('str').to_list()
                data_types = df_cube['ch_type'].to_list()
                filters = ['raw'] * len(data_types)
                channel_names = [[e] for e in df_cube['electrode'].to_list()]
                #if cnt % 1000 == 1: print(f'{cnt}/{n_x * n_y * n_z}')
                job_name = f'slight_{cnt}'
                output_log = f'slight_{cnt}.out'
                error_log = f'slight_{cnt}.err'
                
                # LAUNCH
                cmd = f'python {os.path.join(path2code, script_name)}'
                for patient, data_type, filt, channel_name in zip(patients, data_types, filters, channel_names):
                    cmd += f' --patient {patient} --data-type {data_type} --filter {filt} --channel-name {" ".join(channel_name)}'
                cmd += f' --level sentence_onset'
                cmd += f' --smooth {smooth} --decimate {decimate}'
                cmd += f' --coords {round(x, 2)} {round(y, 2)} {round(z, 2)} --side {side}'
                cmd += f' --comparison-name {comparison_name}'
                if cluster:
                    cmd = f"echo {cmd} | qsub -q {queue} -N {job_name} -l walltime={walltime} -o {os.path.join(path2code, logdir, output_log)} -e {os.path.join(path2code, logdir, error_log)}"
                
                # print(cmd)  
                os.system(cmd)