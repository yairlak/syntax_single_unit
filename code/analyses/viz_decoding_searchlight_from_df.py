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
import nibabel as nib
from utils.viz import create_nifti_statmap, get_time_indices
from utils.viz import update_dataframe, generate_html_brain, compute_vis_aud_intersection
from nilearn import datasets
from nilearn import image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--tmin', default=-0.5, type=float,
                    help='time slice [sec]')
parser.add_argument('--tmax', default=0.5, type=float,
                    help='time slice [sec]')
parser.add_argument('--smooth', default=50, type=int,
                    help='gaussian width in [msec]')
parser.add_argument('--decimate', default=50, type=int)
parser.add_argument('--side-half', default=5.0, type=float, help='Side of cube in mm')
parser.add_argument('--stride', default=5.0, type=float, help='Stride')

# STATS
parser.add_argument('--mean-max', default='mean', choices = ['mean', 'max'],
                    help='Take mean or max of scores across the time domain')
parser.add_argument('--filter-criterion', default='pval_min',
                    choices = ['fdr_min', 'pval_min'],
                    help='filter maps based on fdr or, alternatively, based on pvalue (without fdr)')
parser.add_argument('--alpha-map', default=0.05, type=float,
                    help='for FDR correction]')
parser.add_argument('--alpha-fdr', default=0.05, type=float,
                    help='for FDR correction]')

# QUERY
parser.add_argument('--comparison-name', default='embedding_vs_long_end', #'embedding_vs_long_3rd_word', #'unacc_unerg_dec',
                    help='See Utils/comparisons.py')
parser.add_argument('--comparison-name-test', default=None,
                    help='See Utils/comparisons.py')
parser.add_argument('--block-train', choices=['auditory', 'visual'],
                    default='auditory',
                    help='Block type is added to the query in the comparison')
parser.add_argument('--block-test', choices=['auditory', 'visual'],
                    default='auditory',
                    help='Block type is added to the query in the comparison')
parser.add_argument('--path2output', default='../../Output/decoding')
parser.add_argument('--timewise', default=False, action='store_true')
args = parser.parse_args()

# NILEARN STUFF
mask_ICV = '../../../templates/mask_ICV.nii'
fsaverage = datasets.fetch_surf_fsaverage()
img_ICV = nib.load(mask_ICV)

# LOAD DATAFRAME
args.dummy = '*'
args2fname = ['comparison_name', 'dummy',
              'smooth', 'decimate',
              'side_half', 'stride']   # List of args
fn_pattern = dict2filename(args.__dict__, '_', args2fname, '', True)
print(f'Loading DataFrame: df_{fn_pattern}.json')
df = pd.read_json(os.path.join(args.path2output, 'df_' + fn_pattern + '.json'))
df = df[['scores', 'pvals', 'times', 'args', 'block_train', 'block_test']]

# LOAD COORDINATES
path2coords = '../../Data/UCLA/MNI_coords/'
fn_coords = 'electrode_locations.csv'
df_coords = pd.read_csv(os.path.join(path2coords, fn_coords))
coords_montage = df_coords[['MNI_x', 'MNI_y', 'MNI_z']].values

# GET INDEX OF TIME SLICE
times = np.asarray(df['times'].tolist()[0])
IX_tmin, IX_tmax, IXs_slices = get_time_indices(times, 
                                                args.tmin, args.tmax,
                                                args.timewise)


for IX in IXs_slices: # LOOP OVER TIME RANGE 
    d_dfs = {}
    for block_train in ['visual', 'auditory']: # LOOP OVER BOTH BLOCKS
        d_dfs[block_train] = {}
        for block_test in ['visual', 'auditory']: # LOOP OVER BOTH BLOCKS
            
            if block_test == block_train:
                query = f'block_train=="{block_train}" & block_test!=block_test'
            else:
                query = f'block_train=="{block_train}" & block_test=="{block_test}"'
            d_dfs[block_train][block_test] = df.copy().query(query)

            if IX=='All':
                t = None
                fn_fig = f'../../Figures/viz_brain/{fn_pattern}_{block_train}_{block_test}_tmin_{args.tmin}_tmax_{args.tmax}_query_{args.filter_criterion}_{args.alpha_map}_{args.mean_max}_t_all'
            else:
                IX_tmin, IX_tmax = IX, IX
                t = times[IX]
                fn_fig = f'../../Figures/viz_brain/{fn_pattern}_{block_train}_{block_test}_query_{args.filter_criterion}_{args.alpha_map}_{args.mean_max}_t_{t:.2f}'
               
                
            d_dfs[block_train][block_test] = update_dataframe(d_dfs[block_train][block_test],
                                                              IX_tmin, IX_tmax,
                                                              mean_max=args.mean_max,
                                                              alpha_fdr=args.alpha_fdr)
                
            # FILTER DATAFRAME BASE ON CRITERION
            if args.filter_criterion == 'pval_min':
                query = f'pvals_min<{args.alpha_map}'
            elif args.filter_criterion == 'fdr_min':
                query = f'pvals_fdr_min<{args.alpha_fdr}'
            else:
                raise('Unknown filter criterion')
            d_dfs[block_train][block_test] = d_dfs[block_train][block_test].query(query)
            
    
            if d_dfs[block_train][block_test].empty:
                print(f'Empty dataframe for {block_train}/{block_test}')
                continue
            
            
            #######################################
            # HTML BRAIN
            #######################################
            
            print(f'Plotting HTML brain {block_train}/{block_test} {t}')
            fig_html = generate_html_brain(d_dfs[block_train][block_test],
                                           coords_montage)
            fig_html.save_as_html(fn_fig + '.html')
            
            #######################################
            # GLASS BRAIN
            #######################################
            
            print(f'Plotting glass brain {block_train}/{block_test} {t}')
            smoothing = 5
            coords_results = d_dfs[block_train][block_test][['x', 'y', 'z']].values
            nimg = create_nifti_statmap(coords_results,
                                        args.side_half,
                                        d_dfs[block_train][block_test]['scores_statistic'],
                                        img_ICV,
                                        baseline=0.5)
            
            nimg = image.smooth_img(nimg, smoothing)
            fig_glass = plotting.plot_glass_brain(nimg,
                                                  resampling_interpolation='nearest',
                                                  cmap='RdBu_r',
                                                  colorbar=True,
                                                  vmin=0.3,
                                                  vmax=0.7,
                                                  # threshold=0.5,
                                                  display_mode='lzr')
            fig_glass.add_markers(coords_montage, 'k', marker_size=0.01) 
            fig_glass._cbar.set_label('AAA')
            fig_glass.savefig(fn_fig + '_glass.png')
            fig_glass.close()
    
            #######################################
            # INFLATED BRAIN
            #######################################
            # REGENERATE NIFTI WITH ZERO BASELINE
            nimg = create_nifti_statmap(coords_results,
                                        args.side_half,
                                        d_dfs[block_train][block_test]['scores_statistic'],
                                        img_ICV,
                                        baseline=0)
            
            nimg = image.smooth_img(nimg, smoothing)
            
            if args.block_train == args.block_test:
                if args.block_train == 'visual':
                    cmap = 'RdBu_r'
                else:
                    cmap = 'RdBu'
            else:
                cmap = 'PuBu_r'
            for inflate in [False, True]:
                print(f'Plotting inflated {inflate} brain {block_train}/{block_test} {t}')
                fig_surf, axs = plotting.plot_img_on_surf(nimg,
                                                          hemispheres=['left', 'right'],
                                                          views=['lateral', 'medial', 'ventral'],
                                                          cmap=cmap,
                                                          inflate=inflate,
                                                          threshold=0.5,
                                                          vmax=1)
                fig_surf.savefig(fn_fig + f'_surf_inflate_{inflate}.png')
                plt.close(fig_surf)
                
    for intersection_type in ['within_modalities', 'across_modalities']:
        if IX=='All':
            fn_fig = f'../../Figures/viz_brain/{fn_pattern}_intersection_{intersection_type}_tmin_{args.tmin}_tmax_{args.tmax}_query_{args.filter_criterion}_{args.alpha_map}_{args.mean_max}_t_all'
        else:
            fn_fig = f'../../Figures/viz_brain/{fn_pattern}_intersection_{intersection_type}_query_{args.filter_criterion}_{args.alpha_map}_{args.mean_max}_t_{t:.2f}'
        
        if intersection_type == 'within_modalities':
            df_vis_and_aud = compute_vis_aud_intersection(d_dfs['visual']['visual'],
                                                          d_dfs['auditory']['auditory'])
        elif intersection_type == 'across_modalities':
            df_vis_and_aud = compute_vis_aud_intersection(d_dfs['visual']['auditory'],
                                                          d_dfs['auditory']['visual'])
            
        if df_vis_and_aud.empty:
            continue
        #######################################
        # HTML BRAIN
        #######################################
        
        print(f'Plotting HTML brain {block_train}/{block_test} {t}')
        fig_html = generate_html_brain(df_vis_and_aud,
                                       coords_montage)
        fig_html.save_as_html(fn_fig + '.html')
        
        #######################################
        # GLASS BRAIN
        #######################################
        
        print(f'Plotting glass brain {block_train}/{block_test} {t}')
        smoothing = 5
        coords_results = df_vis_and_aud[['x', 'y', 'z']].values
        nimg = create_nifti_statmap(coords_results,
                                    args.side_half,
                                    df_vis_and_aud['scores_statistic'],
                                    img_ICV,
                                    baseline=0.5)
        
        nimg = image.smooth_img(nimg, smoothing)
        fig_glass = plotting.plot_glass_brain(nimg,
                                              resampling_interpolation='nearest',
                                              cmap='RdBu_r',
                                              colorbar=True,
                                              vmin=0.3,
                                              vmax=0.7,
                                              # threshold=0.5,
                                              display_mode='lzr')
        fig_glass.add_markers(coords_montage, 'k', marker_size=0.01) 
        fig_glass._cbar.set_label('AAA')
        fig_glass.savefig(fn_fig + '_glass.png')
        fig_glass.close()
        
        #######################################
        # INFLATED BRAIN
        #######################################
        # REGENERATE NIFTI WITH ZERO BASELINE
        nimg = create_nifti_statmap(coords_results,
                                    args.side_half,
                                    df_vis_and_aud['scores_statistic'],
                                    img_ICV,
                                    baseline=0)
        
        nimg = image.smooth_img(nimg, smoothing)
        
        if args.block_train == args.block_test:
            if args.block_train == 'visual':
                cmap = 'RdBu_r'
            else:
                cmap = 'RdBu'
        else:
            cmap = 'PuBu_r'
        for inflate in [False, True]:
            print(f'Plotting inflated {inflate} brain {block_train}/{block_test} {t}')
            fig_surf, axs = plotting.plot_img_on_surf(nimg,
                                                      hemispheres=['left', 'right'],
                                                      views=['lateral', 'medial', 'ventral'],
                                                      cmap=cmap,
                                                      inflate=inflate,
                                                      threshold=0.5,
                                                      vmax=1)
            fig_surf.savefig(fn_fig + f'_surf_inflate_{inflate}.png')
            plt.close(fig_surf)
