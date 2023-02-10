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
import nibabel as nib
from utils.viz import voxel_masker
from nilearn import surface
from nilearn import datasets
from nilearn import plotting
from nilearn import image

parser = argparse.ArgumentParser()
parser.add_argument('--tmin', default=0.1, type=float,
                    help='time slice [sec]')
parser.add_argument('--tmax', default=0.8, type=float,
                    help='time slice [sec]')
parser.add_argument('--alpha', default=0.05, type=float,
                    help='for FDR correction]')
parser.add_argument('--smooth', default=50, type=int,
                    help='gaussian width in [msec]')
parser.add_argument('--decimate', default=50, type=int)
parser.add_argument('--side-half', default=6, type=float, help='Side of cube in mm')
parser.add_argument('--mean-max', default='mean', choices = ['mean', 'max'],
                    help='Take mean or max of scores across the time domain')
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


print('Loading DataFrame from json file...')
args2fname = ['comparison_name', 'comparison_name_test',
              'block_train', 'block_test',
              'smooth', 'decimate',
              'side']   # List of args
fn_pattern = dict2filename(args.__dict__, '_', args2fname, '', True)
fn_pattern = 'dec_quest_len2_auditory_dec_quest_len2_visual'
fn_pattern = 'dec_quest_len2_None_auditory_None_50_50_8'
fn_pattern = 'embedding_vs_long_visual_macro'
fn_pattern = 'embedding_vs_long_auditory_macro'
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
df['pvals'] = df.apply(lambda row: get_time_slice(row, 'pvals', IX_min, IX_max), axis=1)
# df['pvals_max'] = df.apply(lambda row: np.max(row['pvals']) if row['pvals'] else None, axis=1)
# df['reject_fdr'], df['pvals_fdr'] = fdr_correction(df['pvals'],
#                                                    alpha=args.alpha,
#                                                    method='indep')
# df['reject_fdr_max'] = df.apply(lambda row: np.max(row['pvals_fdr']) if row['pvals'] else None, axis=1)

# df_query = df.query('reject_fdr==True')
# df_query = df.query('pvals_mean<0.01')

query = 'scores_statistic>0.55'
df_query = df.query(query)
assert not df_query.empty, f'Query resulted in an empty DataFrame: "{query}"'

print('Plotting...')
# VIZ WITH NILEARN
coords_results = df_query[['x', 'y', 'z']].values
colors = df_query.apply(lambda row: cm.RdBu_r(row['scores_statistic']), axis=1).values
colors = np.array(list(colors))

marker_sizes = [16] * colors.shape[0]


# LOAD COORDINATES
path2coords = '../../Data/UCLA/MNI_coords/'
fn_coords = 'electrode_locations.csv'
df_coords = pd.read_csv(os.path.join(path2coords, fn_coords))

coords_montage = df_coords[['MNI_x', 'MNI_y', 'MNI_z']].values

coords = np.vstack((coords_results, coords_montage))
blacks = np.zeros_like(coords_montage)
blacks = np.hstack((blacks, np.ones((blacks.shape[0], 1))))
colors = np.vstack((colors, blacks))
marker_sizes += [2] * blacks.shape[0]
view = plotting.view_markers(coords, colors,
                             marker_size=marker_sizes) 

# view.open_in_browser()
fn = f'../../Figures/viz_brain/{fn_pattern}_tmin_{args.tmin}_tmax_{args.tmax}_query_{query}_{args.mean_max}'
view.save_as_html(fn + '.html')



mask_ICV = '../../../templates/mask_ICV.nii'
fsaverage = datasets.fetch_surf_fsaverage()
img_ICV = nib.load(mask_ICV)

masker, value, [a, b, c] = voxel_masker(coords_results, img_ICV, plot=False)

mask_img = masker.mask_img
mask_img = image.smooth_img(mask_img, 5.)
fig_glass = plotting.plot_glass_brain(mask_img, display_mode='lzr')
fig_glass.add_markers(coords_montage, 'k', marker_size=0.01) 

fig_glass.savefig(fn + '_glass.png')


# texture = surface.vol_to_surf(mask_img, fsaverage.pial_right)
# fig = plotting.plot_surf_stat_map(
#     fsaverage.infl_right, texture, hemi='right',
#     title='Surface right hemisphere', colorbar=True,
#     threshold=1e-2, bg_map=fsaverage.sulc_right
# )
# texture = surface.vol_to_surf(mask_img, fsaverage.pial_left)
# fig = plotting.plot_surf_stat_map(
#     fsaverage.infl_left, texture, hemi='left',
#     title='Surface right hemisphere', colorbar=True,
#     threshold=1e-2, bg_map=fsaverage.sulc_right
# )

for inflate in [False, True]:
    fig_surf, axs = plotting.plot_img_on_surf(mask_img,
                                              views=['lateral', 'medial'],
                                              hemispheres=['left', 'right'],
                                              colorbar=True,
                                              inflate=inflate,
                                              threshold=1e-3)
    fig_surf.savefig(fn + f'_surf_inflate_{inflate}.png')
