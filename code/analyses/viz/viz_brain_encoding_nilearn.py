#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import nibabel as nib
import mne
import matplotlib.pyplot as plt
import pandas as pd
from mne.stats import fdr_correction
from nilearn import plotting  
import nibabel as nib
from helpers import create_nifti_statmap
# from utils.viz import update_dataframe, generate_html_brain, compute_vis_aud_intersection
from nilearn import datasets
from nilearn import image
import matplotlib.pyplot as plt
from mne.stats import fdr_correction
from nilearn.surface import vol_to_surf
from nilearn.surface.surface import _check_mesh

def find_coordinates(row, df_coords, coord):
    ch_name = row['Ch_name']
    probe_name = row['Probe_name']
    data_type = row['data_type']
    patient = row['Patient']
    if str(patient).startswith(('479', '554')):
        patient = int(str(patient)[:3])
    
    if data_type == 'macro':
        query = f'patient == {patient} & ch_name == "{ch_name}" & ch_type == "{data_type}"'
    elif data_type == 'micro':
        query = f'patient == {patient} & electrode.str.contains("_micro-1") & probe_name == "{probe_name}" & ch_type == "{data_type}"'
    elif data_type == 'spike':
        query = f'patient == {patient} & ch_name == "{ch_name}" & ch_type == "{data_type}"'
    
    df_coords = df_coords.query(query)
    
    if len(df_coords) > 1:
        print(df_coords)
        raise()
    elif df_coords.empty:
        return None
    # print(coord, df_coords[f'MNI_{coord}'].values[0])
    return df_coords[f'MNI_{coord}'].values[0]

# LOAD COORDINATES
path2coords = '../../../Data/UCLA/MNI_coords/'
fn_coords = 'electrode_locations.csv'
df_coords = pd.read_csv(os.path.join(path2coords, fn_coords))
# coords_montage = df_coords[['MNI_x', 'MNI_y', 'MNI_z']].values
coords_montage = {}
coords_montage['micro'] = df_coords[df_coords['ch_type']=='micro'][['MNI_x', 'MNI_y', 'MNI_z']].values
coords_montage['macro'] = df_coords[df_coords['ch_type']=='macro'][['MNI_x', 'MNI_y', 'MNI_z']].values

surf_mesh = _check_mesh('fsaverage5')

# block = 'visual'
# data_type = 'micro'
d = {}
filt = 'raw'
query = 'reject_fdr == True & scores > 0'
query_merge = '(reject_fdr_x == True | reject_fdr_y == True ) & (scores_x > 0 | scores_y > 0)'

features = ['full', 'syntax', 'glove', 'lexicon', 'orthography', 'phonemes', 'boundaries']
# features = ['syntax', 'glove', 'lexicon', 'orthography', 'phonemes', 'boundaries']
# features = ['phonemes', 'orthography']
features = ['full']

for feature in features:
    for block in ['auditory', 'visual']:
        d[block] = {}
        for data_type in ['macro', 'micro']:
            d[block][data_type] = None
            fn_trf_results = f'../../../Output/encoding_models/encoding_results_{data_type}_{filt}_{block}_decimate_50_smooth_50_patients_479_11_479_25_482_499_502_505_513_515_538_540_541_543_545_549_551_552_553_554_4_554_13_510_530_539_544_556.json'
            
            print(f'Loading {data_type} {block} data: {fn_trf_results}')
            df_encoding = pd.read_json(fn_trf_results)
            df_encoding = df_encoding[df_encoding['Feature'] == feature]
            if df_encoding.empty:
                print(f'Dataframe is empry {data_type} {block} {feature}')
                continue
            print('Extracting coordinates')
            for coord in ['x', 'y', 'z']:
                df_encoding[coord] = df_encoding.apply(lambda row: find_coordinates(row, df_coords, coord),
                                                       axis=1)
            
            # remove channels without coordinates
            df_encoding = df_encoding[df_encoding['x'].notna()]
            
            
            if feature == 'full':
                ps = df_encoding[f'ps_full_{block}_sentence_trf'].values
                df_encoding['scores'] = df_encoding[f'rs_full_{block}_sentence_trf'].values
                ps = df_encoding[f'ps_full_{block}_sentence_trf'].values # Again, after filtering
                vmax = 0.3
                threshold = df_encoding['scores'][df_encoding['scores']>0].min()
            else:
                ps = df_encoding[f'ps_feature_{block}_sentence_trf'].values
                df_encoding['scores'] = df_encoding[f'rs_full_{block}_sentence_trf'].values - df_encoding[f'rs_feature_{block}_sentence_trf'].values
                ps = df_encoding[f'ps_feature_{block}_sentence_trf'].values # Again, after filtering
                # vmax = np.percentile(df_encoding['scores'][df_encoding['scores']>0], 90)
                vmax = 0.01
                if feature in ['phonemes', 'orthography']:
                    vmax = 0.03
                threshold = np.percentile(df_encoding['scores'][df_encoding['scores']>0], 10)
                
            # FDR
            alpha = 0.01
            df_encoding['reject_fdr'], df_encoding['ps_fdr'] = \
                                fdr_correction(ps, alpha=alpha, method='indep')
            
            
            # del df_encoding       
            d[block][data_type] = df_encoding.copy()
            
            df_encoding = df_encoding.query(query)
            
            coords_results = df_encoding[['x', 'y', 'z']].values
            
            # PLOT
            print('Plotting')
            mask_ICV = '../../../../templates/mask_ICV.nii'
            img_ICV = nib.load(mask_ICV)
            
            #######################################
            # GLASS BRAIN
            #######################################
            nimg = create_nifti_statmap(coords_results,
                                        1,
                                        df_encoding['scores'],
                                        img_ICV,
                                        baseline=0)
            smoothing = 3
            nimg = image.smooth_img(nimg, smoothing)
            
            if block == 'auditory':
                cmap = 'Blues'
            elif block == 'visual':
                cmap = 'Reds'
            
            fig_glass = plotting.plot_glass_brain(nimg,
                                                  resampling_interpolation='nearest',
                                                  cmap=cmap,
                                                  colorbar=True,
                                                  vmin=0,
                                                  vmax=vmax,
                                                  # threshold=0.5,
                                                  display_mode='lzry')
            fig_glass.add_markers(coords_montage[data_type], 'k', marker_size=0.01) 
            fig_glass._cbar.set_label('r')
            fn_fig = f'../../../Figures/encoding/{feature}_{data_type}_{filt}_{block}'
            fig_glass.savefig(fn_fig + '_glass.png')
            fig_glass.close()
            
            #######################################
            # INFLATED BRAIN
            #######################################
            # REGENERATE NIFTI WITH ZERO BASELINE
            nimg = create_nifti_statmap(coords_results,
                                        2,
                                        df_encoding['scores'],
                                        img_ICV,
                                        baseline=0)
            smoothing = 5
            nimg = image.smooth_img(nimg, smoothing)
            
            
            
            if block == 'auditory':
                cmap = 'RdBu'
            elif block == 'visual':
                cmap = 'RdBu_r'
                # cmap = 'PuBu_r'
                
            for inflate in [False, True]:
                print(f'Plotting inflated {inflate} brain')
                fig_surf, axs = plotting.plot_img_on_surf(nimg,
                                                          hemispheres=['left', 'right'],
                                                          views=['lateral', 'medial', 'ventral'],
                                                          cmap=cmap,
                                                          inflate=inflate,
                                                          threshold=threshold,
                                                          vmax=vmax)
                fig_surf.savefig(fn_fig + f'_surf_inflate_{inflate}.png')
                plt.close(fig_surf)
                
            # del nimg, axs, fig_glass, fig_surf, mask_ICV, img_ICV, inflate
            # del scores, coords_results
        #######################################
        # GLASS BRAIN
        #######################################
        mask_ICV = '../../../../templates/mask_ICV.nii'
        img_ICV = nib.load(mask_ICV)
        
        if (d[block]['micro'] is None) or (not d[block]['macro'] is None):
            continue
        
        coords_results = np.vstack((d[block]['micro'].query(query)[['x', 'y', 'z']].values,
                                    d[block]['macro'].query(query)[['x', 'y', 'z']].values))
        scores = np.hstack((d[block]['micro'].query(query)['scores'],
                            d[block]['macro'].query(query)['scores']))
        
        if feature == 'full':
            vmax = 0.3
            threshold = scores[scores>0].min()
        else:
            # vmax = np.percentile(scores[scores>0], 90)
            vmax = 0.01
            if feature in ['phonemes', 'orthography']:
                vmax = 0.03
            threshold = np.percentile(scores[scores>0], 10)
        
        
        
        nimg = create_nifti_statmap(coords_results,
                                    1,
                                    scores,
                                    img_ICV,
                                    baseline=0)
        smoothing = 3
        nimg = image.smooth_img(nimg, smoothing)
        
        if block == 'auditory':
            cmap = 'Blues'
        elif block == 'visual':
            cmap = 'Reds'
        
        fig_glass = plotting.plot_glass_brain(nimg,
                                              resampling_interpolation='nearest',
                                              cmap=cmap,
                                              colorbar=True,
                                              vmin=0,
                                              vmax=vmax,
                                              # threshold=0.5,
                                              display_mode='lzry')
        
        coords_montage_all = np.vstack((coords_montage['micro'], coords_montage['macro']))
        fig_glass.add_markers(coords_montage_all, 'k', marker_size=0.01) 
        fig_glass._cbar.set_label('r')
        fn_fig = f'../../../Figures/encoding/{feature}_{filt}_{block}'
        fig_glass.savefig(fn_fig + '_glass.png')
        fig_glass.close()
        
        
        #######################################
        # INFLATED BRAIN
        #######################################
        # REGENERATE NIFTI WITH ZERO BASELINE
        nimg = create_nifti_statmap(coords_results,
                                    2,
                                    scores,
                                    img_ICV,
                                    baseline=0)
        smoothing = 5
        nimg = image.smooth_img(nimg, smoothing)
        
        
        
        if block == 'auditory':
            cmap = 'RdBu'
        elif block == 'visual':
            cmap = 'RdBu_r'
            # cmap = 'PuBu_r'
            
        for inflate in [False, True]:
            print(f'Plotting inflated {inflate} brain')
            fig_surf, axs = plotting.plot_img_on_surf(nimg,
                                                      hemispheres=['left', 'right'],
                                                      views=['lateral', 'medial', 'ventral'],
                                                      cmap=cmap,
                                                      inflate=inflate,
                                                      threshold=threshold,
                                                      vmax=vmax)
            fig_surf.savefig(fn_fig + f'_surf_inflate_{inflate}.png')
            plt.close(fig_surf)
    
        
    df_aud_vis_micro = pd.merge(d['auditory']['micro'], d['visual']['micro'], on='Ch_name', how='inner')
    df_aud_vis_macro = pd.merge(d['auditory']['macro'], d['visual']['macro'], on='Ch_name', how='inner')
    df_aud_vis_micro = df_aud_vis_micro.query(query_merge)
    df_aud_vis_macro = df_aud_vis_macro.query(query_merge)

    df_aud_vis_macro.loc[df_aud_vis_macro['scores_x'] < 0, 'scores_x'] = 0
    df_aud_vis_macro.loc[df_aud_vis_macro['scores_y'] < 0, 'scores_y'] = 0
    df_aud_vis_micro.loc[df_aud_vis_micro['scores_x'] < 0, 'scores_x'] = 0
    df_aud_vis_micro.loc[df_aud_vis_micro['scores_y'] < 0, 'scores_y'] = 0
    
    #######################################
    # GLASS BRAIN
    #######################################
    mask_ICV = '../../../../templates/mask_ICV.nii'
    img_ICV = nib.load(mask_ICV)
    
    if (d[block]['micro'] is None) or (d[block]['macro'] is None):
        continue
    coords_results = np.vstack((df_aud_vis_micro[['x_x', 'y_x', 'z_x']].values,
                                df_aud_vis_macro[['x_x', 'y_x', 'z_x']].values))
    scores_micro = (df_aud_vis_micro['scores_y'].values - df_aud_vis_micro['scores_x'].values)
    scores_macro = (df_aud_vis_macro['scores_y'].values - df_aud_vis_macro['scores_x'].values)
    scores = np.concatenate((scores_micro, scores_macro))
    
    if feature == 'full':
        vmax = None
        vmax = 0.1
        threshold = scores[scores>0].min()
    else:
        # vmax = np.percentile(scores[scores>0], 90)
        vmax = None
        if feature in ['phonemes', 'orthography']:
            vmax = 0.03
        # vmax = 0.5
        threshold = np.percentile(scores[scores>0], 10)
    
    
    
    nimg = create_nifti_statmap(coords_results,
                                1,
                                scores,
                                img_ICV,
                                baseline=0)
    smoothing = 3
    nimg = image.smooth_img(nimg, smoothing)
    
    if block == 'auditory':
        cmap = 'Blues'
    elif block == 'visual':
        cmap = 'Reds'
    
    fig_glass = plotting.plot_glass_brain(nimg,
                                          resampling_interpolation='nearest',
                                          cmap=cmap,
                                          colorbar=True,
                                          vmin=0,
                                          vmax=vmax,
                                          # threshold=0.5,
                                          display_mode='lzry')
    
    coords_montage_all = np.vstack((coords_montage['micro'], coords_montage['macro']))
    fig_glass.add_markers(coords_montage_all, 'k', marker_size=0.01) 
    fig_glass._cbar.set_label('r')
    fn_fig = f'../../../Figures/encoding/{feature}_{filt}'
    fig_glass.savefig(fn_fig + '_glass.png')
    fig_glass.close()
    
    
    #######################################
    # INFLATED BRAIN
    #######################################
    # REGENERATE NIFTI WITH ZERO BASELINE
    nimg = create_nifti_statmap(coords_results,
                                2,
                                scores,
                                img_ICV,
                                baseline=0)
    smoothing = 5
    nimg = image.smooth_img(nimg, smoothing)
    
    
    
    if block == 'auditory':
        cmap = 'RdBu'
    elif block == 'visual':
        cmap = 'RdBu_r'
        # cmap = 'PuBu_r'
        
    alpha= 1
    for inflate in [False, True]:
        print(f'Plotting inflated {inflate} brain')
        fig_surf, axs = plotting.plot_img_on_surf(nimg,
                                                  hemispheres=['left', 'right'],
                                                  views=['lateral', 'medial', 'ventral'],
                                                  cmap=cmap,
                                                  inflate=inflate,
                                                  threshold=threshold,
                                                  vmax=vmax,
                                                  **{'alpha':alpha})
        fig_surf.savefig(fn_fig + f'_surf_inflate_{inflate}.png')
        plt.close(fig_surf)