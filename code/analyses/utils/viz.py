#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:46:41 2022

@author: yl254115
"""
import os
import pandas as pd
import nibabel as nib
import numpy as np
from nilearn import input_data, plotting
from nilearn.image import new_img_like
from mne.stats import fdr_correction
import matplotlib.cm as cm

def voxel_masker(coords_list, img, plot=False, **kwargs_masker): 
    """Returns: - masker for given coordinates - value for input image at this location - voxel coordinates
        coords_list: list
         each element is a tuple for the three coordinates (xyz)
    
    """ 
    if type(img)==str:
        img = nib.load(img)
    
    affine = img.affine[:3, :3] 
    translation = img.affine[:3, 3]
    
    new_data = np.zeros(img.get_fdata().shape)

    for coords in coords_list:
        data_coords = np.matmul(np.linalg.inv(affine), np.array(coords) - translation)
        
        a,b,c = np.apply_along_axis(lambda x: np.round(x, 0), 0, data_coords).astype(int)
        
        value = img.get_fdata()[a, b, c]
        
        new_data[a,b,c] = 1
        
    masker = input_data.NiftiMasker(new_img_like(img, new_data))
    
    masker.set_params(**kwargs_masker)
    
    masker.fit()
    
    if plot: plotting.plot_glass_brain(masker.mask_img, display_mode='lzry')
    
    plotting.show()
    
    return masker, value, [a, b, c]


def create_nifti_statmap(coords_list, side_half, values, img, baseline=0): 
    """Returns: - masker for given coordinates - value for input image at this location - voxel coordinates
        coords_list: list
         each element is a tuple for the three coordinates (xyz)
    
    """ 
    if type(img)==str:
        img = nib.load(img)
    
    affine = img.affine[:3, :3] 
    translation = img.affine[:3, 3]
    
    new_data = np.ones(img.get_fdata().shape) * baseline

    for coords, value in zip(coords_list, values):
   
        data_coords_min = np.matmul(np.linalg.inv(affine), np.array(coords-side_half) - translation)
        data_coords_max = np.matmul(np.linalg.inv(affine), np.array(coords+side_half) - translation)
        
        a_min,b_min,c_min = np.apply_along_axis(lambda x: np.round(x, 0), 0, data_coords_min).astype(int)
        a_max,b_max,c_max = np.apply_along_axis(lambda x: np.round(x, 0), 0, data_coords_max).astype(int)
        
        
        for i in range(min(a_min, a_max), max(a_min, a_max)+1):
            for j in range(min(b_min, b_max), max(b_min, b_max)+1):
                for k in range(min(c_min, c_max), max(c_min, c_max)+1):
                   new_data[i,j,k] = value
        
    nimg = new_img_like(img, new_data)
    
   
    
    return nimg


def get_time_indices(times, tmin, tmax, timewise):
    
    times_tmin = np.abs(times - tmin)
    IX_tmin = np.argmin(times_tmin)
    if tmax:
        times_tmax = np.abs(times - tmax)
        IX_tmax = np.argmin(times_tmax)
    else:
        IX_tmax = None
        
    if timewise: # timepoint by timepoint or all slices
        IXs_slices = range(IX_tmin, IX_tmax+1)
    else: # All slices
        IXs_slices = ['All']
    
    
    return IX_tmin, IX_tmax, IXs_slices



def get_time_slice(row, key, IX_min, IX_max):
    # GET SCORES AND PVALS FOR TIME SLICE
    values = row[key]
    if (IX_min in range(len(values))) and (IX_max in range(len(values))):
        if IX_max:
            return values[IX_min:IX_max+1]
        else:    
            return values[IX_min]
    else:
        return None
    
    
def get_coords(row, dim):
    args = row['args']
    if 'coords' in args.keys():
        return args['coords'][dim-1]
    else:
        return None
    

def update_dataframe(df, IX_tmin, IX_tmax, mean_max='mean', alpha_fdr=0.05):
    
    # EXTRACT COORDINATES SEPARATELY FOR XYZ
    df['x'] = df.apply(lambda row: get_coords(row, 1), axis=1)
    df['y'] = df.apply(lambda row: get_coords(row, 2), axis=1)
    df['z'] = df.apply(lambda row: get_coords(row, 3), axis=1)
    df = df.drop('args', axis=1)
    
    # EXTRACT VALUES WITHIN TIME RANGE (IX_tmin to IX_tmax)
    df['times'] = df.apply(lambda row: get_time_slice(row, 'times', IX_tmin, IX_tmax), axis=1)
    df['scores'] = df.apply(lambda row: get_time_slice(row, 'scores', IX_tmin, IX_tmax), axis=1)
    df['pvals'] = df.apply(lambda row: get_time_slice(row, 'pvals', IX_tmin, IX_tmax), axis=1)
    
    
    # COMPUTE STATISTIC FOR BRAIN MAPS (MEAN OR MAX SCORE)
    if mean_max == 'mean':
        df['scores_statistic'] = df.apply(lambda row: np.mean(row['scores']) if row['scores'] else None, axis=1)
    elif mean_max == 'max':
        df['scores_statistic'] = df.apply(lambda row: np.max(row['scores']) if row['scores'] else None, axis=1)
    
    # FDR
    pvals = np.stack(df['pvals'])
    reject_fdr, pvals_fdr = fdr_correction(pvals,
                                           alpha=alpha_fdr,
                                           method='indep')
    
    df['reject_fdr'] = reject_fdr.tolist()
    df['pvals_fdr'] = pvals_fdr.tolist()
    
    # MIN PVAL ACROSS TIME
    df['pvals_min'] = df.apply(lambda row: np.min(row['pvals']) if row['pvals'] else None, axis=1)
    df['pvals_fdr_min'] = df.apply(lambda row: np.min(row['pvals_fdr']) if row['pvals_fdr'] else None, axis=1)

    return df


def generate_html_brain(df, coords_montage):
    # VIZ WITH NILEARN
    scores_max = df['scores_statistic'].max()
    coords_results = df[['x', 'y', 'z']].values
    colors = df.apply(lambda row: cm.RdBu_r((row['scores_statistic']-0.5)/(scores_max-0.5)), axis=1).values
    colors = np.array(list(colors))
    
    marker_sizes = [16] * colors.shape[0]
    
    coords = np.vstack((coords_results, coords_montage))
    blacks = np.zeros_like(coords_montage)
    blacks = np.hstack((blacks, np.ones((blacks.shape[0], 1))))
    colors = np.vstack((colors, blacks))
    marker_sizes += [2] * blacks.shape[0]
    view = plotting.view_markers(coords, colors,
                                 marker_size=marker_sizes) 
    
    return view


def compute_vis_aud_intersection(df1, df2):
    df1 = df1[['x', 'y', 'z', 'scores_statistic']]
    df2 = df2[['x', 'y', 'z', 'scores_statistic']]
    
    df_intersection = pd.merge(df1, df2, how='inner', on=['x', 'y', 'z'])
    df_intersection['scores_statistic'] = df_intersection[['scores_statistic_x', 'scores_statistic_y']].mean(axis=1)
    df_intersection.dropna(inplace=True)
    return df_intersection