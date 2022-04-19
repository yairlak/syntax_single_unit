#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:26:27 2022

@author: yl254115
"""

import pandas as pd
import matplotlib.pyplot as plt
from mne.stats import fdr_correction

# LOAD ENCODING RESULTS
fn = '../../../Output/encoding_models/encoding_results_decimate_50_smooth_50.json'
df = pd.read_json(fn)

# MISC
alpha = 0.05

#################################
# PLOT r-sentence DISTRIBUTIONS #
#################################
features = list(set(df['Feature']))
for feature in features:
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    for i_dt, data_type in enumerate(['micro', 'macro', 'spike']):
        df_curr = df.loc[df['data_type'] == data_type]
        df_curr = df_curr[df_curr['Feature'] == feature]
        df_curr = df_curr[df_curr['filter'] == 'raw'] # COMPARE RAWS ONLY
        for i_block, block in enumerate(['visual', 'auditory']):
            if (block == 'visual' and feature=='phonology') or \
               (block == 'auditory' and feature=='orthography'):
                   continue
            rs_sentence_feature = df_curr[f'rs_feature_{block}_sentence_trf']
            rs_sentence_full = df_curr[f'rs_full_{block}_sentence_trf']
            # STATS
            ps_sentence_full = df_curr[f'ps_full_{block}_sentence_trf']
            reject_fdr, pvals_fdr = fdr_correction(ps_sentence_full,
                                                   alpha=alpha,
                                                   method='indep')
            # POSITIVE r only
            IXs_positive_r = rs_sentence_full > 0
            
            rs_sentence_full = rs_sentence_full[IXs_positive_r & reject_fdr]
            rs_sentence_feature = rs_sentence_feature[IXs_positive_r & reject_fdr]
            
            if feature == 'full':
                drs_sentence = rs_sentence_full
            else:
                drs_sentence = rs_sentence_full - rs_sentence_feature
            color = {'visual':'r', 'auditory':'b'}[block]
            axs[i_block, i_dt].hist(drs_sentence, color=color, bins=50)
            axs[i_block, i_dt].set_xlabel('Brain score (r)', fontsize=16)
            xmax = 1 if feature=='full' else 0.1
            axs[i_block, i_dt].set_xlim([0, xmax])
            
            if i_block == 0:
                axs[i_block, i_dt].set_title(data_type.capitalize(), fontsize=16)
    
    plt.suptitle(feature.capitalize(), fontsize=24)
    fn_fig = f'../../../Figures/encoding_models/dist_r_sentence_{feature}_TRF_all_data_types.png'
    plt.savefig(fn_fig)
    print(f'Figure saved to: {fn_fig}')
    plt.close(fig)
