#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:26:27 2022

@author: yl254115
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.stats import fdr_correction
from statsmodels.stats.weightstats import ttest_ind
from mne.stats import spatio_temporal_cluster_1samp_test

# LOAD ENCODING RESULTS
fn = '../../../Output/encoding_models/encoding_results_decimate_50_smooth_50.json'
df = pd.read_json(fn)

# CONFIG
word_or_sentence = 'word'
fdr = False
regress_method = 'trf'

def compute_stats_for_r_sentence(row, block):
    rs_sentence_full = row[f'rs_full_{block}_sentence_per_split_trf']
    rs_sentence_feature = row[f'rs_feature_{block}_sentence_per_split_trf']
    if rs_sentence_full and rs_sentence_feature:
        tstat, pval, df = ttest_ind(rs_sentence_full, rs_sentence_feature)
        return pval
    else:
        return None
    
    
def compute_stats_for_r_word(row, block, method):    
    """Statistical test applied across CV splits"""
    
    rs_word_full = row[f'rs_full_{block}_word_per_split_{method}']
    rs_word_feature = row[f'rs_feature_{block}_word_per_split_{method}']
    if rs_word_full and rs_word_feature:
        dr = np.asarray([np.asarray(curr_split_rs_word_full) - np.asarray(curr_split_rs_word_feature) \
                         for (curr_split_rs_word_full, curr_split_rs_word_feature) in \
                         zip(rs_word_full, rs_word_feature)])
        
        # stats function report p_value for each cluster
        dr = dr[:, :, None] if dr.ndim == 2 else dr
        _, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(dr, out_type='mask',
                                                                      n_permutations=1000, n_jobs=-1,
                                                                      verbose=False, tail=1, seed=42)
    
    
        p_values_ = np.ones_like(dr[0]).T
        for cluster, pval in zip(clusters, p_values):
            p_values_[cluster.T] = pval
    
        return np.squeeze(p_values_).T
    else:
        return None



def get_mask_fdr(df, alpha, word_or_sentence, method):
    if word_or_sentence == 'sentence':
        ps_dr_sentence_trf = df_curr[f'ps_dr_sentence_{block}_trf']
        reject_fdr_dr, _ = fdr_correction(ps_dr_sentence_trf,
                                          alpha=alpha,
                                          method='indep')
    elif word_or_sentence == 'word':
        method = 'trf' # !
        df_curr[f'ps_dr_word_{block}_{method}'] = df_curr.apply(lambda row: compute_stats_for_r_word(row, block, method),
                                                                axis=1)
        ps_dr_word = np.stack(df_curr[f'ps_dr_word_{block}_{method}'].values)
        reject_fdr_ps_dr_word, _ = fdr_correction(ps_dr_word,
                                                  alpha=alpha,
                                                  method='indep')
        
        reject_fdr_dr = np.asarray([l.any() for l in reject_fdr_ps_dr_word])
    return reject_fdr_dr
    
    
    
for block in ['visual', 'auditory']:
    df[f'ps_dr_sentence_{block}_trf'] = df.apply(lambda row: compute_stats_for_r_sentence(row, block),
                                                  axis=1)


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
               
            # SENTENCE STATS
            rs_sentence_feature = df_curr[f'rs_feature_{block}_sentence_trf']
            rs_sentence_full = df_curr[f'rs_full_{block}_sentence_trf']
            
            # SIGNIFICANCE OF FULL MODEL
            ps_sentence_full = df_curr[f'ps_full_{block}_sentence_trf']
            reject_fdr_sentence_full, _ = fdr_correction(ps_sentence_full,
                                                         alpha=alpha,
                                                         method='indep')
            
            # SIGNIFICANCE OF DELTA r
           
            if feature != 'full' and fdr:
                reject_fdr_dr = get_mask_fdr(df_curr, alpha,
                                             word_or_sentence, regress_method)
            else:
                reject_fdr_dr = np.ones(len(df_curr))
               
            # POSITIVE r only
            IXs_positive_r = rs_sentence_full > 0
            rs_sentence_full = rs_sentence_full[IXs_positive_r & reject_fdr_sentence_full & reject_fdr_dr]
            rs_sentence_feature = rs_sentence_feature[IXs_positive_r & reject_fdr_sentence_full & reject_fdr_dr]
            
            
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
    fn_fig = f'../../../Figures/encoding_models/dist_r_sentence_{feature}_TRF_all_data_types_fdr_{fdr}.png'
    plt.savefig(fn_fig)
    print(f'Figure saved to: {fn_fig}')
    plt.close(fig)
