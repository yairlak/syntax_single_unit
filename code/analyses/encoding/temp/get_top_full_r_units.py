#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:26:27 2022

@author: yl254115
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.stats import fdr_correction
from utils_encoding import get_mask_fdr


parser = argparse.ArgumentParser()
parser.add_argument('--word-or-sentence', default='sentence')
parser.add_argument('--regress_method', default='trf',
                    choices=['trf', 'evoked'])
parser.add_argument('--negative_values_too', default=False, action='store_true')
parser.add_argument('--alpha', default=0.1)
args = parser.parse_args()

# LOAD ENCODING RESULTS
patients = ['479_11', '479_25', '482', '499', '502', '505', '510', '513', '515',
            '530', '538', '539', '540', '541', '543', '544', '549', '551']
data_type_filters = ['micro_raw', 'micro_high-gamma',
                     'macro_raw', 'macro_high-gamma',
                     'spike_raw']

assert not (args.word_or_sentence=='sentence' and args.regress_method=='evoked')

features = ['phonology', 'orthography', 'lexicon', 'semantics', 'syntax', 'position', 'full']
for feature in features:
    n_subplots = len(data_type_filters)
    fig, axs = plt.subplots(2, n_subplots, figsize=(5*n_subplots, 5))
    for i_dt, data_type_filter in enumerate(data_type_filters):
        data_type, filt = data_type_filter.split('_')
        # LOAD RESULTS
        fn = f'../../../Output/encoding_models/encoding_results_'
        fn += f'{data_type}_{filt}_decimate_50_smooth_50_patients_{"_".join(patients)}.json'
        df_data_type_filt = pd.read_json(fn)    
        df_data_type_filt = df_data_type_filt[df_data_type_filt['Feature'] == feature]

        for i_block, block in enumerate(['visual', 'auditory']):
            if (block == 'visual' and feature=='phonology') or \
               (block == 'auditory' and feature=='orthography'):
                   continue
            print(feature, data_type, filt, block)
            
            # Mask FDR correction, full model, sentence-level
            field_mask = f'reject_fdr_r_full_{args.regress_method}_{args.word_or_sentence}_0.005_{block}'
            df_data_type_filt_block_fdr = df_data_type_filt[df_data_type_filt[field_mask]]
                
            
            if feature == 'full':
                field = f'rs_full_{block}_{args.word_or_sentence}_{args.regress_method}'
                scores = np.asarray(df_data_type_filt_block_fdr[field])
                xlabel = 'Brain score'
            else:
                reject_fdr_dr, ps_fdr_dr, drs = get_mask_fdr(df_data_type_filt_block_fdr,
                                                             block,
                                                             args.regress_method,
                                                             args.word_or_sentence,
                                                             'dr',
                                                             alpha=args.alpha)
                scores = np.asarray(drs)[reject_fdr_dr]
                xlabel = 'Brain-score difference'
                
            color = {'visual':'r', 'auditory':'b'}[block]
            axs[i_block, i_dt].hist(scores, color=color, bins=50)
            axs[i_block, i_dt].set_xlabel(xlabel, fontsize=16)
            (xmin, xmax) = (-0.1, 1) if feature=='full' else (-0.05, 0.2)
            axs[i_block, i_dt].set_xlim([-0.1, xmax])
            
            if i_block == 0:
                axs[i_block, i_dt].set_title(f'{data_type.capitalize()} ({filt})', fontsize=16)
    
    plt.suptitle(feature.capitalize(), fontsize=24)
    fn_fig = f'../../../Figures/encoding_models/dist_r_'
    fn_fig += f'{args.word_or_sentence}_{feature}_{args.regress_method}_'
    fn_fig += f'fdr_{args.alpha}.png'
    plt.savefig(fn_fig)
    plt.subplots_adjust(top=0.8)
    print(f'Figure saved to: {fn_fig}')
    plt.close(fig)
