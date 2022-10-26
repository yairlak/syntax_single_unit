#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:02:02 2022

@author: yair
"""

import os
import argparse
import pandas as pd
import numpy as np
from utils.utils import dict2filename
from decoding.utils import get_channel_names_from_cube_center

parser = argparse.ArgumentParser()
parser.add_argument('--tmin', default=-1, type=float,
                    help='time slice [sec]')
parser.add_argument('--tmax', default=1, type=float,
                    help='time slice [sec]')
parser.add_argument('--alpha', default=0.05, type=float,
                    help='for FDR correction]')
parser.add_argument('--smooth', default=50, type=int,
                    help='gaussian width in [msec]')
parser.add_argument('--decimate', default=50, type=int)
parser.add_argument('--side', default=8, type=float, help='Side of cube in mm')
parser.add_argument('--mean-max', default='mean', choices = ['mean', 'max'],
                    help='Take mean or max of scores across the time domain')
parser.add_argument('--top-k', default=10,
                    help='Channel names for the top k scores values.')
# QUERY
parser.add_argument('--comparison-name', default='grammatical_number',
                    help='See Utils/comparisons.py')
parser.add_argument('--comparison-name-test', default=None,
                    help='See Utils/comparisons.py')
parser.add_argument('--block-train', choices=['auditory', 'visual'],
                    default='auditory',
                    help='Block type is added to the query in the comparison')
parser.add_argument('--block-test', choices=['auditory', 'visual'],
                    default=None,
                    help='Block type is added to the query in the comparison')
parser.add_argument('--path2output', default='../../Output/decoding')
args = parser.parse_args()


print('Loading DataFrame from json file...')
args2fname = ['comparison_name', 'comparison_name_test',
              'block_train', 'block_test',
              'smooth', 'decimate',
              'side']   # List of args
fn_pattern = dict2filename(args.__dict__, '_', args2fname, '', True)
fn_pattern = 'dec_quest_len2_auditory_dec_quest_len2_visual'
# fn_pattern = 'dec_quest_len2_None_auditory_None_50_50_8'
fn_pattern = 'embedding_vs_long_visual_macro'
# fn_pattern = 'embedding_vs_long_auditory_macro'
fn_pattern = 'embedding_vs_long_auditory_embedding_vs_long_visual'
fn_pattern = 'number_all_auditory_number_all_visual_macro'
if fn_pattern == 'dec_quest_len2_None_auditory_None_50_50_8':
    hack = True
else:
    hack = False
df = pd.read_json(os.path.join(args.path2output, 'df_' + fn_pattern + '.json'))
df = df[['scores', 'pvals', 'times', 'args']]


print('Getting coordinates, scores and p-values...')
def get_coords(row, dim, hack):
    args = row['args']
    if hack:
        if dim == 1:
            return args['x']
        if dim == 2:
            return args['y']
        if dim == 3:
            return args['z']
    else:
        if 'coords' in args.keys():
            return args['coords'][dim-1]
        else:
            return None
        

df['x'] = df.apply(lambda row: get_coords(row, 1, hack), axis=1)
df['y'] = df.apply(lambda row: get_coords(row, 2, hack), axis=1)
df['z'] = df.apply(lambda row: get_coords(row, 3, hack), axis=1)
df = df.drop('args', axis=1)

# GET INDEX OF TIME SLICE
times = np.asarray(df['times'][0])
times_tmin = np.abs(times - args.tmin)
IX_min = np.argmin(times_tmin)
if args.tmax:
    times_tmax = np.abs(times - args.tmax)
    IX_max = np.argmin(times_tmax)
else:
    IX_max = None

# GET SCORES AND PVALS FOR TIME SLICE
def get_time_slice(row, key, IX_min, IX_max):
    values = row[key]
    if (IX_min in range(len(values))) and (IX_max in range(len(values))):
        if IX_max:
            return values[IX_min:IX_max+1]
        else:    
            return values[IX_min]
    else:
        return None
    
df['times'] = df.apply(lambda row: get_time_slice(row, 'times', IX_min, IX_max), axis=1)
df['scores'] = df.apply(lambda row: get_time_slice(row, 'scores', IX_min, IX_max), axis=1)
if args.mean_max == 'mean':
    df['scores_statistic'] = df.apply(lambda row: np.mean(row['scores']) if row['scores'] else None, axis=1)
elif args.mean_max == 'max':
    df['scores_statistic'] = df.apply(lambda row: np.max(row['scores']) if row['scores'] else None, axis=1)

df_sorted = df.sort_values('scores_statistic', ascending=False)
df_sorted = df_sorted.head(args.top_k)
print(df_sorted)

path2coords = '../../Data/UCLA/MNI_coords/'
fn_coords = 'electrode_locations.csv'
df_coords = pd.read_csv(os.path.join(path2coords, fn_coords))

data_types = ['macro']
filt = 'raw'
for i_row, row in df_sorted.iterrows():
    x, y, z = row['x'], row['y'], row['z']
    patients, channel_names, probe_names, ch_nums = \
        get_channel_names_from_cube_center(df_coords,
                                           x, y, z,
                                           args.side, data_types)
    print(patients, channel_names)
    for patient, channel_name in zip(patients, channel_names):
        cmd = f'python plot_ERP_trialwise.py --patient {patient}'
        cmd += f' --data-type {data_types[0]} --filter {filt}'
        cmd += f' --comparison-name {args.comparison_name}'
        cmd += f' --channel-name {channel_name}'
        print(cmd)