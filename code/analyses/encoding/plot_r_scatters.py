#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:26:27 2022

@author: yl254115
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne.stats import fdr_correction

# LOAD ENCODING RESULTS
fn = '../../../Output/encoding_models/encoding_results_decimate_50_smooth_50.json'
df = pd.read_json(fn)

# MISC
alpha = 0.05
   
############################
# PLOT r-sentence SCATTERS #
############################
features = list(set(df['Feature'])-set(['phonology', 'orthography']))
for feature in features:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    for i_dt, data_type in enumerate(['micro', 'macro', 'spike']):
        df_curr = df.loc[df['data_type'] == data_type]
        df_curr = df_curr[df_curr['Feature'] == feature]
        df_curr = df_curr[df_curr['filter'] == 'raw'] # COMPARE RAWS ONLY
        
        rs_sentence_feature_visual = df_curr[f'rs_feature_visual_sentence_trf']
        rs_sentence_full_visual = df_curr[f'rs_full_visual_sentence_trf']
        rs_sentence_feature_auditory = df_curr[f'rs_feature_auditory_sentence_trf']
        rs_sentence_full_auditory = df_curr[f'rs_full_auditory_sentence_trf']
        
        if feature == 'full':
            drs_sentence_visual = rs_sentence_full_visual
            drs_sentence_auditory = rs_sentence_full_auditory
        else:
            drs_sentence_visual = rs_sentence_full_visual - rs_sentence_feature_visual
            drs_sentence_auditory = rs_sentence_full_auditory - rs_sentence_feature_auditory
        
        axs[i_dt].scatter(drs_sentence_visual, drs_sentence_auditory,
                          s=1, c='k')
        axs[i_dt].set_xlabel('Visual', fontsize=16)
        axs[i_dt].set_ylabel('Auditory', fontsize=16)
        xmax = 1 if feature=='full' else 0.08
        axs[i_dt].set_xlim([0, xmax])
        axs[i_dt].set_ylim([0, xmax])
        
        axs[i_dt].set_title(data_type.capitalize(), fontsize=16)
        axs[i_dt].set_aspect('equal', adjustable='box')
        axs[i_dt].plot([0, 1], [0, 1], transform=axs[i_dt].transAxes, ls='--', color='k', lw=2)
    
    plt.suptitle(feature.capitalize(), fontsize=24)
    fn_fig = f'../../../Figures/encoding_models/scatter_r_sentence_{feature}_TRF_all_data_types.png'
    plt.savefig(fn_fig)
    print(f'Figure saved to: {fn_fig}')
    plt.close(fig)
    
    
for feature in features:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    for i_dt, data_type in enumerate(['micro', 'macro', 'spike']):
        df_curr = df.loc[df['data_type'] == data_type]
        df_curr = df_curr[df_curr['Feature'] == feature]
        df_curr = df_curr[df_curr['filter'] == 'raw'] # COMPARE RAWS ONLY
        
        rs_word_feature_visual = df_curr[f'rs_feature_visual_word_trf'].apply(np.max)
        rs_word_full_visual = df_curr[f'rs_full_visual_word_trf'].apply(np.max)
        rs_word_feature_auditory = df_curr[f'rs_feature_auditory_word_trf'].apply(np.max)
        rs_word_full_auditory = df_curr[f'rs_full_auditory_word_trf'].apply(np.max)
        
        if feature == 'full':
            drs_word_visual = rs_word_full_visual
            drs_word_auditory = rs_word_full_auditory
        else:
            drs_word_visual = rs_word_full_visual - rs_word_feature_visual
            drs_word_auditory = rs_word_full_auditory - rs_word_feature_auditory
        
        axs[i_dt].scatter(drs_word_visual, drs_word_auditory,
                          s=1, c='k')
        axs[i_dt].set_xlabel('Visual', fontsize=16)
        axs[i_dt].set_ylabel('Auditory', fontsize=16)
        xmax = 1 if feature=='full' else 0.08
        axs[i_dt].set_xlim([0, xmax])
        axs[i_dt].set_ylim([0, xmax])
        
        axs[i_dt].set_title(data_type.capitalize(), fontsize=16)
        axs[i_dt].set_aspect('equal', adjustable='box')
        axs[i_dt].plot([0, 1], [0, 1], transform=axs[i_dt].transAxes,
                       ls='--', color='k', lw=2)
    
    plt.suptitle(feature.capitalize(), fontsize=24)
    fn_fig = f'../../../Figures/encoding_models/scatter_r_word_max_{feature}_TRF_all_data_types.png'
    plt.savefig(fn_fig)
    print(f'Figure saved to: {fn_fig}')
    plt.close(fig)