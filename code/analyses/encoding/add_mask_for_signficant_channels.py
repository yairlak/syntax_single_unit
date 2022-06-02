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
parser.add_argument('--negative_values_too', default=False,
                    action='store_true')
parser.add_argument('--alpha', default=0.005)
args = parser.parse_args()

# LOAD ENCODING RESULTS
patients = ['479_11', '479_25', '482', '499', '502', '505', '510', '513', '515',
            '530', '538', '539', '540', '541', '543', '544', '549', '551']
data_type_filters = ['micro_raw', 'micro_high-gamma',
                     'macro_raw', 'macro_high-gamma',
                     'spike_raw']

# The 'evoked' method has no sentence-level r score. 
assert not (args.word_or_sentence=='sentence' and args.regress_method=='evoked')

# Find channels, whose r score for the full model is not significant,
# or negative. 

for i_dt, data_type_filter in enumerate(data_type_filters):
    data_type, filt = data_type_filter.split('_')
    print(data_type, filt)
    
    # LOAD RESULTS
    fn = f'../../../Output/encoding_models/encoding_results_'
    fn += f'{data_type}_{filt}_decimate_50_smooth_50_patients_{"_".join(patients)}.json'
    df = pd.read_json(fn)    
    df_full = df[df['Feature'] == 'full']

    # VISUAL BLOCKS
    reject_fdr_r_full_vis, _, rs_full_vis = get_mask_fdr(df_full, 'visual',
                                                         args.regress_method,
                                                         args.word_or_sentence,
                                                         'r',
                                                         alpha=args.alpha)
    mask_vis = reject_fdr_r_full_vis * (rs_full_vis>0)
    
    # AUDITORY BLOCKS
    reject_fdr_r_full_aud, _, rs_full_aud = get_mask_fdr(df_full, 'auditory',
                                                         args.regress_method,
                                                         args.word_or_sentence,
                                                         'r',
                                                         alpha=args.alpha)
    mask_aud = reject_fdr_r_full_aud * (rs_full_aud>0)
    
    print(f'Significant channels (visual): {sum(mask_vis)}/{len(df_full)}')  
    print(f'Significant channels (auditory): {sum(mask_aud)}/{len(df_full)}')
    
    # Create a list with all significant channels,
    # each represented by a tuple (ch_name, patient)
    channels_to_keep_viz = list(zip(df_full[mask_vis]['Ch_name'], df_full[mask_vis]['Patient']))
    channels_to_keep_aud = list(zip(df_full[mask_aud]['Ch_name'], df_full[mask_aud]['Patient']))
    mask_viz_df, mask_aud_df = [], []
    for _, row in df.iterrows():
        if (row['Ch_name'], row['Patient']) in channels_to_keep_viz:
            mask_viz_df.append(True)
        else:
            mask_viz_df.append(False)
        
        if (row['Ch_name'], row['Patient']) in channels_to_keep_aud:
            mask_aud_df.append(True)
        else:
            mask_aud_df.append(False)
    df[f'reject_fdr_r_full_{args.regress_method}_{args.word_or_sentence}_{args.alpha}_visual'] = mask_viz_df
    df[f'reject_fdr_r_full_{args.regress_method}_{args.word_or_sentence}_{args.alpha}_auditory'] = mask_aud_df
        
        
    df.to_json(fn)
    print(f'File saved to (override): {fn}')
