#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:26:27 2022

@author: yl254115
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.stats import fdr_correction
from utils_encoding import get_mask_fdr, convert_pvalue_to_asterisks
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--data-type-filter', default='spike_raw')
parser.add_argument('--top-k', default=100)
parser.add_argument('--word-or-sentence', default='sentence')
parser.add_argument('--regress_method', default='trf',
                    choices=['trf', 'evoked'])
parser.add_argument('--negative_values_too', default=False, action='store_true')
parser.add_argument('--alpha', default=0.05)
args = parser.parse_args()

# 
path2figures = '../../../Figures/encoding_models/channel_profiles/'
os.makedirs(path2figures, exist_ok=True)

# LOAD ENCODING RESULTS
patients = ['479_11', '479_25', '482', '499', '502', '505', '510', '513', '515',
            '530', '538', '539', '540', '541', '543', '544', '549', '551']

assert not (args.word_or_sentence=='sentence' and args.regress_method=='evoked')


data_type, filt = args.data_type_filter.split('_')

# LOAD RESULTS
path2output = f'../../../Output/encoding_models/'
fn_results =  os.path.join(path2output,
                           f'encoding_results_{data_type}_{filt}_decimate_50_smooth_50_patients_{"_".join(patients)}.json')
df_data_type_filt = pd.read_json(fn_results) 

for i_block, block in enumerate(['auditory', 'visual']):
    # REJECT FDR BASED ON FULL MODEL, SENTENCE-LEVEL
    field_mask = f'reject_fdr_r_full_{args.regress_method}_{args.word_or_sentence}_0.005_{block}'
    df_data_type_filt_block_fdr = df_data_type_filt[df_data_type_filt[field_mask]]   
    df_data_type_filt_block_fdr_full = df_data_type_filt[df_data_type_filt['Feature'] == 'full']
    # SORT DATAFRAME
    field_score = f'rs_full_{block}_sentence_trf'
    df_data_type_filt_block_fdr_full = \
        df_data_type_filt_block_fdr_full.sort_values(field_score,
                                                     ascending=False)
    
    df_data_type_filt_block_fdr_full_top_k = df_data_type_filt_block_fdr_full.head(args.top_k)
    
    #  EXPORT SORTED DATAFRAME TO HTML
    df_html = df_data_type_filt_block_fdr_full[['Patient',
                                                'Ch_name',
                                                f'rs_full_{block}_sentence_trf']].reset_index(drop=True)
    fn_html = f'Top_scores_{data_type}_{filt}_{block}_{args.regress_method}_{args.word_or_sentence}.html'
    fn_html = os.path.join(path2output, fn_html)
    df_html.to_html(fn_html)
    print(f"HTML saved to: {fn_html}")
    
    
    features = ['phonology', 'orthography', 'lexicon', 'semantics', 'syntax', 'position']
    for i_ch, row in df_data_type_filt_block_fdr_full_top_k.iterrows():
        ch_name = row['Ch_name']
        patient = row['Patient']
        print(patient, ch_name)
        # MASK TO GET A SINGLE CHANNEL
        mask = (df_data_type_filt_block_fdr['Ch_name'] == ch_name) * \
               (df_data_type_filt_block_fdr['Patient'] == patient)
        df_data_type_filt_block_fdr_top_k_curr = \
            df_data_type_filt_block_fdr[mask]
        # GET r FULL SENTENCE TRF
        r_block_full_sentence_trf = df_data_type_filt_block_fdr_top_k_curr[f'rs_full_{block}_sentence_trf'].values[0]
        # GET DELTA r PER FEATURE
        df_for_figure = []
        for feature in features:
            if (block == 'visual' and feature=='phonology') or \
               (block == 'auditory' and feature=='orthography'):
               continue
            df_curr_channel_feature = df_data_type_filt_block_fdr_top_k_curr[df_data_type_filt_block_fdr_top_k_curr['Feature']==feature]
            assert len(df_curr_channel_feature) == 1
            _, p, dr = get_mask_fdr(df_curr_channel_feature,
                                    block,
                                    args.regress_method,
                                    args.word_or_sentence,
                                    'dr',
                                    alpha=args.alpha)
            df_for_figure.append([patient, ch_name, feature, dr[0], p])
            # print(patient, ch_name, feature, dr[0])
        df_for_figure = pd.DataFrame(df_for_figure, columns = ['Patient', 'Ch_name', 'Feature', 'Score', 'p-value'])
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.barplot(data=df_for_figure, x='Score', y='Feature',
                    ax=ax, orient='h')
        ax.set_title(f'Patient {patient}, Channel {ch_name} (r={r_block_full_sentence_trf:1.2f})')
        ymax = np.max((0.1, df_for_figure['Score'].max()))
        ax.set_xlim([0, ymax])
        ax.tick_params(axis='both', labelsize=24)
        ax.set_xlabel('Feature importance', fontsize=30)
        # ADD ASTERISKS
        for i, row in df_for_figure.iterrows():
            row['p-value'][0]
            y_pos = row['Score'] + 0.002
            ax.text(y_pos, i, convert_pvalue_to_asterisks(row['p-value'][0]),
                    fontsize=24)
        
        
        fn_fig = f'channel_profile_{data_type}_{filt}_{patient}_{ch_name}_{block}.png'
        fig.savefig(os.path.join(path2figures, fn_fig))
        plt.close(fig)
        