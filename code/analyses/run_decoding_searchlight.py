#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:19:32 2022

@author: yair
"""

import os
import argparse
import pandas as pd
import numpy as np
from MNI_coords import UtilsCoords

parser = argparse.ArgumentParser()
parser.add_argument('--block-train', default='auditory')
parser.add_argument('--block-test', default='auditory')
parser.add_argument('--cluster', action='store_true', default=False)
parser.add_argument('--launch', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--print', action='store_true', default=False)
args = parser.parse_args()

# CUBE
side_half = 5 # Half the size of the side of cube
stride = side_half # could be, e.g., side_half/2

# DATA
smooth = 25
decimate = 50
k_bins = 4

#'dec_quest_len2 embedding_vs_long_end embedding_vs_long_3rd_word number_subject number_verb unacc_unerg_dec'
comparison_name = 'number_subject'

# DATA TYPES
isMacro = True
isMicro = True
isSpike = True

# BLOCK TYPES
#blocks_train = ['auditory', 'visual']
#blocks_test = ['auditory', 'visual']

# CLUSTER
queue = 'Nspin_long'
walltime = '72:00:00'

# LOAD COORDINATES
path2code =  '/neurospin/unicog/protocols/intracranial/syntax_single_unit/code/analyses/'
#path2code = '/home/yair/projects/syntax_single_unit/code/analyses'
logdir = 'logs'
script_name = 'decoding.py'

# LOAD COORDINATES
path2coords = '../../Data/UCLA/MNI_coords/'
fn_coords = 'electrode_locations.csv'
df = pd.read_csv(os.path.join(path2coords, fn_coords), index_col=0)
df = df.query('patient != 504')

x_min, x_max = df['MNI_x'].min(), df['MNI_x'].max()
y_min, y_max = df['MNI_z'].min(), df['MNI_y'].max()
z_min, z_max = df['MNI_z'].min(), df['MNI_y'].max()

n_x, n_y, n_z = len(np.arange(x_min, x_max+side_half, stride)), len(np.arange(y_min, y_max+side_half, stride)), len(np.arange(z_min, z_max+side_half, stride))

# FOR DEBUG
if args.debug:
    x_min, y_min, z_min = -44.47, -16.7, 7.94
    x_max = x_min+1
    y_max = y_min+1
    z_max = z_min+1

cnt, cnt_run = 0, 0
n_channels = []
for x in np.arange(x_min, x_max+side_half, stride):
    for y in np.arange(y_min, y_max+side_half, stride):
        for z in np.arange(z_min, z_max+side_half, stride):
            
            #if cnt % 10 == 0: print(f'{cnt+1}/{n_x * n_y * n_z}')
            cnt += 1
            
            # PICK CHANNELS IN CUBE
            df_cube = UtilsCoords.pick_channels_by_cube(df,
                                                        (x, y, z),
                                                        side_half=side_half,
                                                        isMacro=isMacro,
                                                        isMicro=isMicro,
                                                        isSpike=isSpike,
                                                        isStim=False)
            # GET DATA AND DECODE
            if not df_cube.empty:
                cnt_run += 1
                #if 'micro' in df_cube.ch_type.values:
                #if isSpike:
                #    df_cube = UtilsCoords.add_spike_rows(df_cube)
                #print(x, y, z, side_half)
                #print(df_cube.to_string())
                patients = df_cube['patient'].astype('str').to_list()
                data_types = df_cube['ch_type'].to_list()
                filters = ['raw'] * len(data_types)
                channel_names = [[e] for e in df_cube['ch_name'].to_list()]
                job_name = f'slight_{cnt}'
                output_log = f'slight_{cnt}.out'
                error_log = f'slight_{cnt}.err'

                # LAUNCH
                cmd = f'python3 {os.path.join(path2code, script_name)}'
                for patient, data_type, filt, channel_name in zip(patients, data_types, filters, channel_names):
                    cmd += f' --patient {patient} --data-type {data_type} --filter {filt} --channel-name {" ".join(channel_name)}'
                cmd += f' --level sentence_onset'
                cmd += f' --smooth {smooth} --decimate {decimate}'
                cmd += f' --coords {round(x, 2)} {round(y, 2)} {round(z, 2)} --side-half {side_half} --stride {stride}'
                cmd += f' --comparison-name {comparison_name}'
                cmd += f' --k-bins {k_bins}'
                cmd += f' --block-train {args.block_train}'
                if args.block_test:
                    cmd += f' --block-test {args.block_test}'
                
                if args.cluster:
                    cmd = f"echo {cmd} | qsub -q {queue} -N {job_name} -l walltime={walltime} -o {os.path.join(path2code, logdir, output_log)} -e {os.path.join(path2code, logdir, error_log)}"
                if args.launch:
                    os.system(cmd)
                if not args.cluster or args.print:
                    print(cmd)
print(f'Number of non-empty cubes: {cnt_run}/{cnt}')
