#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:26:27 2022

@author: yl254115
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne.stats import fdr_correction

# LOAD ENCODING RESULTS
patients = ['479_11', '479_25', '482', '499', '502', '505', '510', '513', '515',
            '530', '538', '539', '540', '541', '543', '544', '549', '551']
data_type_filters = ['micro_raw', 'micro_high-gamma',
                     'macro_raw', 'macro_high-gamma',
                     'spike_raw']
features = ['full', 'position', 'lexicon', 'semantics', 'syntax']
alpha = 0.05
   
############################
# PLOT r-sentence SCATTERS #
############################

for feature in features:
    n_subplots = len(data_type_filters)
    fig_sentence, axs_sentence = plt.subplots(1, n_subplots,
                                              figsize=(5*n_subplots, 5))
    fig_word, axs_word = plt.subplots(1, n_subplots,
                                      figsize=(5*n_subplots, 5))
    for i_dt, data_type_filter in enumerate(data_type_filters):
        data_type, filt = data_type_filter.split('_')
        print(data_type, filt, feature)
        # LOAD RESULTS
        fn = f'../../../Output/encoding_models/encoding_results_{data_type}_{filt}_decimate_50_smooth_50_patients_{"_".join(patients)}.json'
        df = pd.read_json(fn)    
        df_curr = df[df['Feature'] == feature]
        
        # SENTENCE
        rs_sentence_feature_visual = df_curr[f'rs_feature_visual_sentence_trf']
        rs_sentence_full_visual = df_curr[f'rs_full_visual_sentence_trf']
        rs_sentence_feature_auditory = df_curr[f'rs_feature_auditory_sentence_trf']
        rs_sentence_full_auditory = df_curr[f'rs_full_auditory_sentence_trf']
        # WORD
        rs_word_feature_visual = df_curr[f'rs_feature_visual_word_trf'].apply(np.max)
        rs_word_full_visual = df_curr[f'rs_full_visual_word_trf'].apply(np.max)
        rs_word_feature_auditory = df_curr[f'rs_feature_auditory_word_trf'].apply(np.max)
        rs_word_full_auditory = df_curr[f'rs_full_auditory_word_trf'].apply(np.max)
        
        if feature == 'full':
            # SENTENCE
            drs_sentence_visual = rs_sentence_full_visual
            drs_sentence_auditory = rs_sentence_full_auditory
            # WORD
            drs_word_visual = rs_word_full_visual
            drs_word_auditory = rs_word_full_auditory
        else:
            # SENTENCE
            drs_sentence_visual = rs_sentence_full_visual - rs_sentence_feature_visual
            drs_sentence_auditory = rs_sentence_full_auditory - rs_sentence_feature_auditory
            # WORD
            drs_word_visual = rs_word_full_visual - rs_word_feature_visual
            drs_word_auditory = rs_word_full_auditory - rs_word_feature_auditory
            
        # SENTENCE
        axs_sentence[i_dt].scatter(drs_sentence_visual, drs_sentence_auditory,
                                   s=1, c='k')
        axs_sentence[i_dt].set_xlabel('Visual', fontsize=16)
        axs_sentence[i_dt].set_ylabel('Auditory', fontsize=16)
        xmax = 1 if feature=='full' else 0.08
        axs_sentence[i_dt].set_xlim([0, xmax])
        axs_sentence[i_dt].set_ylim([0, xmax])
        
        axs_sentence[i_dt].set_title(data_type.capitalize(), fontsize=16)
        axs_sentence[i_dt].set_aspect('equal', adjustable='box')
        axs_sentence[i_dt].plot([0, 1], [0, 1], transform=axs_sentence[i_dt].transAxes,
                                ls='--', color='k', lw=2)
    
        # WORD
        axs_word[i_dt].scatter(drs_word_visual, drs_word_auditory,
                          s=1, c='k')
        axs_word[i_dt].set_xlabel('Visual', fontsize=16)
        axs_word[i_dt].set_ylabel('Auditory', fontsize=16)
        xmax = 1 if feature=='full' else 0.08
        axs_word[i_dt].set_xlim([0, xmax])
        axs_word[i_dt].set_ylim([0, xmax])
        
        axs_word[i_dt].set_title(data_type.capitalize(), fontsize=16)
        axs_word[i_dt].set_aspect('equal', adjustable='box')
        axs_word[i_dt].plot([0, 1], [0, 1], transform=axs_word[i_dt].transAxes,
                            ls='--', color='k', lw=2)
        
    # SAVE FIGURES
    fig_sentence.suptitle(feature.capitalize() + ' sentence', fontsize=24)
    fn_fig = f'../../../Figures/encoding_models/scatter_r_sentence_{feature}_TRF_all_data_types.png'
    fig_sentence.savefig(fn_fig)
    print(f'Figure saved to: {fn_fig}')
    plt.close(fig_sentence)        

    fig_word.suptitle(feature.capitalize() + ' word', fontsize=24)
    fn_fig = f'../../../Figures/encoding_models/scatter_r_word_max_{feature}_TRF_all_data_types.png'
    fig_word.savefig(fn_fig)
    print(f'Figure saved to: {fn_fig}')
    plt.close(fig_word)