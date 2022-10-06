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
from nilearn import plotting  
import matplotlib.cm as cm
from mne.stats import fdr_correction

parser = argparse.ArgumentParser()
parser.add_argument('--t', default=0.3, type=float,
                    help='time slice [sec]')
parser.add_argument('--alpha', default=0.05, type=float,
                    help='for FDR correction]')
parser.add_argument('--smooth', default=50, type=int,
                    help='gaussian width in [msec]')
parser.add_argument('--decimate', default=50, type=int)
parser.add_argument('--side', default=8, type=float, help='Side of cube in mm')

# QUERY
parser.add_argument('--comparison-name', default='dec_quest_len2',
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


args2fname = ['comparison_name', 'comparison_name_test',
              'block_train', 'block_test',
              'smooth', 'decimate',
              'side']   # List of args
fn_pattern = dict2filename(args.__dict__, '_', args2fname, '', True)
df = pd.read_json(os.path.join(args.path2output, 'df_' + fn_pattern + '.json'))
df = df[['scores', 'pvals', 'times', 'args']]
df['x'] = df.apply(lambda row: row['args']['x'], axis=1)
df['y'] = df.apply(lambda row: row['args']['y'], axis=1)
df['z'] = df.apply(lambda row: row['args']['z'], axis=1)
df = df.drop('args', axis=1)

# GET INDEX OF TIME SLICE
times = np.asarray(df['times'][0])
times_t = np.abs(times - args.t)
IX_min = np.argmin(times_t)

# GET SCORES AND PVALS FOR TIME SLICE
df['times'] = df.apply(lambda row: row['times'][IX_min], axis=1)
df['scores'] = df.apply(lambda row: row['scores'][IX_min], axis=1)
df['pvals'] = df.apply(lambda row: row['pvals'][IX_min], axis=1)

df['reject_fdr'], df['pvals_fdr'] = fdr_correction(df['pvals'],
                                                   alpha=args.alpha,
                                                   method='indep')

# df = df.query('reject_fdr==True')
df = df.query('pvals<0.001')

# VIZ WITH NILEARN
coords = df[['x', 'y', 'z']].values
colors = df.apply(lambda row: cm.RdBu_r(row['scores']), axis=1).values

labels = None
view = plotting.view_markers(coords, colors, marker_labels=labels,
                             marker_size=3) 

# view.open_in_browser()
view.save_as_html(f'../../Figures/viz_brain/{fn_pattern}.html')
