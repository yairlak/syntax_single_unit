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
parser.add_argument('--tmin', default=0, type=float,
                    help='time slice [sec]')
parser.add_argument('--tmax', default=0.7, type=float,
                    help='time slice [sec]')
parser.add_argument('--alpha', default=0.05, type=float,
                    help='for FDR correction]')
parser.add_argument('--smooth', default=50, type=int,
                    help='gaussian width in [msec]')
parser.add_argument('--decimate', default=50, type=int)
parser.add_argument('--side-half', default=5, type=float, help='Side of cube in mm')
parser.add_argument('--mean-max', default='mean', choices = ['mean', 'max'],
                    help='Take mean or max of scores across the time domain')
parser.add_argument('--top-k', default=10,
                    help='Channel names for the top k scores values.')
# QUERY
parser.add_argument('--comparison-name', default='embedding_vs_long_3rd_word',
                    help='See Utils/comparisons.py')
parser.add_argument('--comparison-name-test', default='embedding_vs_long_3rd_word',
                    help='See Utils/comparisons.py')
parser.add_argument('--block-train', choices=['auditory', 'visual'],
                    default='visual',
                    help='Block type is added to the query in the comparison')
parser.add_argument('--block-test', choices=['auditory', 'visual'],
                    default='auditory',
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
fn_pattern = 'dec_quest_len2_None_auditory_visual_50_50_6'

args2fname = ['comparison_name', 'block_train',
              'comparison_name_test', 'block_test']
fn_pattern = dict2filename(args.__dict__, '_', args2fname, '', True)
fn_pattern = f'{fn_pattern}_*_{args.smooth}_{args.decimate}_{args.side_half}'   # List of args


fn_pattern = 'number_subject_*_50_50_5.0'
fn_pattern = 'he_she_*_50_50_5'

df = pd.read_json(os.path.join(args.path2output, 'df_' + fn_pattern + '.json'))
# df = df[['scores', 'pvals', 'times', 'args']]

df = df[['scores', 'pvals', 'times', 'args', 'block_train', 'block_test']]

# QUERY FOR BLOCKS
if args.block_test == args.block_train:
    query = f'block_train=="{args.block_train}" & block_test!=block_test'
else:
    query = f'block_train=="{args.block_train}" & block_test=="{args.block_test}"'
df = df.query(query)

print('Getting coordinates, scores and p-values...')
def get_coords(row, dim):
    args = row['args']
    if 'coords' in args.keys():
        return args['coords'][dim-1]
    else:
        return None
        

df['x'] = df.apply(lambda row: get_coords(row, 1), axis=1)
df['y'] = df.apply(lambda row: get_coords(row, 2), axis=1)
df['z'] = df.apply(lambda row: get_coords(row, 3), axis=1)
df = df.drop('args', axis=1)

# GET INDEX OF TIME SLICE
times = np.asarray(df['times'].tolist()[0])
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

args.timewise = True
if args.timewise:
    IXs_slices = range(IX_min, IX_max+1)
else:
    IXs_slices = [None]
for IX in IXs_slices: 
    if IX is not None:
        IX_min, IX_max = IX, IX
        t = times[IX]
        fn = f'../../Figures/viz_brain/{fn_pattern}_{args.block_train}_{args.block_test}_tmin_{args.tmin}_tmax_{args.tmax}_query_FDR_{args.mean_max}_t_{t:.2f}'
        
    else:
        t = None
        fn = f'../../Figures/viz_brain/{fn_pattern}_{args.block_train}_{args.block_test}_tmin_{args.tmin}_tmax_{args.tmax}_query_FDR_{args.mean_max}_t_all'
    
        
    print(f'Time - {t}')
    df_query = df.copy()    
    df_query['times'] = df_query.apply(lambda row: get_time_slice(row, 'times', IX_min, IX_max), axis=1)
    df_query['scores'] = df_query.apply(lambda row: get_time_slice(row, 'scores', IX_min, IX_max), axis=1)
    if args.mean_max == 'mean':
        df_query['scores_statistic'] = df_query.apply(lambda row: np.mean(row['scores']) if row['scores'] else None, axis=1)
    elif args.mean_max == 'max':
        df_query['scores_statistic'] = df_query.apply(lambda row: np.max(row['scores']) if row['scores'] else None, axis=1)
    
    df_sorted = df_query.sort_values('scores_statistic', ascending=False)
    df_sorted = df_sorted.head(args.top_k)
    print(df_sorted)
    # raise()
    
    path2coords = '../../Data/UCLA/MNI_coords/'
    fn_coords = 'electrode_locations.csv'
    df_coords = pd.read_csv(os.path.join(path2coords, fn_coords))
    
    data_types = ['micro', 'spike', 'macro']
    filt = 'raw'
    for i_row, row in df_sorted.head(2).iterrows():
        x, y, z = row['x'], row['y'], row['z']
        patients, channel_names, probe_names, ch_nums = \
            get_channel_names_from_cube_center(df_coords,
                                               x, y, z,
                                               args.side_half, data_types)
        print(patients, channel_names, row['scores_statistic'])
        for patient, channel_name in zip(patients, channel_names):
            cmd = f'python plot_ERP_trialwise.py --patient {patient}'
            cmd += f' --data-type {data_types[0]} --filter {filt}'
            cmd += f' --comparison-name {args.comparison_name}'
            cmd += f' --channel-name {channel_name}'
            print(cmd)