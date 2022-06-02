#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:48:59 2021

@author: yl254115
"""
import argparse
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import sys
import pickle
from utils.utils import dict2filename
import matplotlib.pyplot as plt
from encoding.models import TimeDelayingRidgeCV
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
# DATA
parser.add_argument('--decimate', default=50, type=int,
                    help='If not empty, decimate data for speed.')
parser.add_argument('--smooth', default=50, type=int,
                    help='Gaussian smoothing in msec')
# MISC
parser.add_argument('--path2output',
                    default=os.path.join('..', '..',
                                         'Output', 'encoding_models'))
parser.add_argument('--path2figures',
                    default=os.path.join('..', '..',
                                         'Figures', 'encoding_models', 'scatters'))
parser.add_argument('--model-type', default='ridge',
                    choices=['ridge', 'lasso', 'ridge_laplacian', 'standard'])
parser.add_argument('--ablation-method', default='remove',
                    choices=['shuffle', 'remove', 'zero'],
                    help='Method used to calcuated feature importance\
                        by reducing/ablating a feature family')
parser.add_argument('--each-feature-value', default=True, action='store_true',
                    help="Evaluate model after ablating each feature value. \
                         If false, ablate all feature values together")


#############
# USER ARGS #
#############

args = parser.parse_args()

def init_dict():
    d = {}
    for block in ['visual', 'auditory']:
        d[block] = {}
        for feature in ['full', 'feature']:
            d[block][feature] = {}
            for level in ['word', 'sentence']:
                d[block][feature][level] = {}
                for model in ['evoked', 'TRF']:
                    d[block][feature][level][model] = {}
    return d
                        


def get_probe_name(channel_name, data_type):
    if data_type == 'micro':
        probe_name = channel_name[4:-1]
    elif data_type == 'macro':
        probe_name = channel_name[1:]
    elif data_type == 'spike':
        probe_name = channel_name
    else:
        print(channel_name, data_type)
        raise('Wrong data type')

    if not probe_name:
        probe_name = 'UNK'
    return probe_name


dict_hemi = {'L':'Left', 'R':'Right'}

print('Collecting results...')


patients="479_11 479_25 482 499 502 505 510 513 515 530 538 539 540 541 543 544 549 551"
#patients="479_11 502"
data_types = ['spike', 'micro', 'macro']
filters = ['raw', 'high-gamma']

args.probe_name = None
for data_type in data_types:
    args.data_type = data_type
    for filt in filters:
        if data_type=='spike' and filt != 'raw':
            continue
        # INIT DATAFRAME FOR EACH DATA-TYPE AND FILTER PAIR
        df = pd.DataFrame()
        feature_groups = "phonology_orthography_lexicon_semantics_syntax".split('_')
        for feature_group in feature_groups:
            for patient in patients.split():
                args.patient = ['patient_' + patient]
                args.filter = filt
                results_evoked = {'auditory':None, 'visual':None}
                results_trf = {'auditory':None, 'visual':None}
                feature_info_evoked = {'auditory':None, 'visual':None}
                feature_info_trf = {'auditory':None, 'visual':None}

                args.feature_list = ['position', feature_group]
                list_args2fname = ['patient', 'data_type', 'filter',
                                   'decimate', 'smooth', 'model_type',
                                   'probe_name', 'ablation_method',
                                   'query_train', 'feature_list', 'each_feature_value']

                found_both_blocks = True
                for block in ['auditory', 'visual']:
                    args.query_train = {'auditory':'block in [2,4,6] and word_length>1',
                                        'visual':'block in [1,3,5] and word_length>1'}[block]
                    args2fname = args.__dict__.copy()
                    # EVOKED
                    fname = 'evoked_' + dict2filename(args2fname, '_', list_args2fname, '', True)
                    try:
                        results_evoked[block], ch_names_evoked, args_evoked, feature_info_evoked[block] = \
                            pickle.load(open(os.path.join(args.path2output, fname + '.pkl'), 'rb'))
                        print(f'Loaded: {fname}')
                    except:
                        print(f'File not found: {args.path2output}/{fname}.pkl')
                        found_both_blocks = False
                        continue
                    
                    # TRF
                    fname = 'TRF_' + dict2filename(args2fname, '_', list_args2fname, '', True)
                    try:
                        results_trf[block], ch_names_trf, args_trf, feature_info_trf[block] = \
                            pickle.load(open(os.path.join(args.path2output, fname + '.pkl'), 'rb'))
                        print(f'Loaded: {fname}')
                    except:
                        print(f'File not found: {args.path2output}/{fname}.pkl')
                        found_both_blocks = False
                        continue
                    
                if (not found_both_blocks) and not (feature_group in ['phonology', 'orthography']):
                    print(f'Skipping patient {patient}, {data_type}, {filt} {feature_group}')
                    continue

                print(f'Patient - {patient}, {data_type}, {filt}, {feature_group}') 
                assert ch_names_evoked==ch_names_trf
                
                # GET TARGET FEATURES:
                feature_info_keys = []
                if results_trf['auditory']:
                    feature_info_keys.extend(list(results_trf['auditory'].keys()))
                if results_trf['visual']:
                    feature_info_keys.extend(list(results_trf['visual'].keys()))
                feature_info_keys = list(set(feature_info_keys) - set(['times_word_epoch'])) # remove duplicates
                
                # FOR EACH CHANNEL AND FEATURE, APPEND A ROW TO DATAFRAME
                for i_ch, ch_name in enumerate(ch_names_trf):
                    probe_name = get_probe_name(ch_name, args.data_type)
                    
                    for feature in feature_info_keys:
                        # TRF
                        curr_results = init_dict()
                        for block_ in ['auditory', 'visual']:
                            for feature_or_full in ['feature', 'full']:
                                if feature_or_full == 'feature':
                                    key = feature
                                else:
                                    key = 'full'
                                if (results_trf[block_] is not None) and (feature in results_trf[block_].keys()):
                                    # TRF word feature
                                    curr_results[block_][feature_or_full]['word']['TRF']['rs'] =  results_trf[block_][key][f'rs_word'][i_ch, :]
                                    curr_results[block_][feature_or_full]['word']['TRF']['ps'] =  results_trf[block_][key][f'ps_word'][i_ch, :]
                                    curr_results[block_][feature_or_full]['word']['TRF']['rs_per_split'] =  [l[i_ch, :] for l in results_trf[block_][key][f'rs_word_per_split']]
                                    curr_results[block_][feature_or_full]['word']['TRF']['ps_per_split'] =  [l[i_ch, :] for l in results_trf[block_][key][f'ps_word_per_split']]
                                    # TRF sentence feature
                                    curr_results[block_][feature_or_full]['sentence']['TRF']['rs'] =  results_trf[block_][key][f'rs_sentence'][i_ch]
                                    curr_results[block_][feature_or_full]['sentence']['TRF']['ps'] =  results_trf[block_][key][f'ps_sentence'][i_ch]
                                    curr_results[block_][feature_or_full]['sentence']['TRF']['rs_per_split'] =  [l[i_ch] for l in results_trf[block_][key][f'rs_sentence_per_split']]
                                    curr_results[block_][feature_or_full]['sentence']['TRF']['ps_per_split'] =  [l[i_ch] for l in results_trf[block_][key][f'ps_sentence_per_split']]
                                    curr_results[block_][feature_or_full]['sentence']['TRF']['rf_per_split'] =  [rf.coef_[i_ch, :, :] for rf in results_trf[block_][key]['rf_sentence_per_split']]
                                    # Evoked word feature
                                    curr_results[block_][feature_or_full]['word']['evoked']['rs'] =  results_evoked[block_][key]['rs_word'][i_ch, :]
                                    curr_results[block_][feature_or_full]['word']['evoked']['ps'] =  results_evoked[block_][key]['ps_word'][i_ch, :]
                                    curr_results[block_][feature_or_full]['word']['evoked']['rs_per_split'] =  [l[i_ch, :] for l in results_evoked[block_][key][f'rs_word_per_split']]
                                    curr_results[block_][feature_or_full]['word']['evoked']['ps_per_split'] =  [l[i_ch, :] for l in results_evoked[block_][key][f'ps_word_per_split']]
                                else:
                                    # TRF word feature
                                    curr_results[block_][feature_or_full]['word']['TRF']['rs'] =  None
                                    curr_results[block_][feature_or_full]['word']['TRF']['ps'] =  None
                                    curr_results[block_][feature_or_full]['word']['TRF']['rs_per_split'] =  None
                                    curr_results[block_][feature_or_full]['word']['TRF']['ps_per_split'] =  None
                                    # TRF sentence feature
                                    curr_results[block_][feature_or_full]['sentence']['TRF']['rs'] =  None
                                    curr_results[block_][feature_or_full]['sentence']['TRF']['ps'] =  None
                                    curr_results[block_][feature_or_full]['sentence']['TRF']['rs_per_split'] =  None
                                    curr_results[block_][feature_or_full]['sentence']['TRF']['ps_per_split'] =  None
                                    curr_results[block_][feature_or_full]['sentence']['TRF']['rf_per_split'] = None 
                                    # Evoked word feature
                                    curr_results[block_][feature_or_full]['word']['evoked']['rs'] =  None
                                    curr_results[block_][feature_or_full]['word']['evoked']['ps'] =  None
                                    curr_results[block_][feature_or_full]['word']['evoked']['rs_per_split'] =  None
                                    curr_results[block_][feature_or_full]['word']['evoked']['ps_per_split'] =  None
                                
                        
                        if probe_name[0] in dict_hemi.keys():
                            hemi = dict_hemi[probe_name[0]]
                        else:
                            hemi = None

                        df = df.append({'data_type':data_type,
                                        'filter':filt,
                                        'Probe_name':probe_name,
                                        'Hemisphere':hemi,
                                        'Ch_name':ch_name,
                                        'Patient':patient, 
                                        'Feature':feature,
                                        # TRF feature word
                                        'rs_feature_visual_word_trf':curr_results['visual']['feature']['word']['TRF']['rs'],
                                        'ps_feature_visual_word_trf':curr_results['visual']['feature']['word']['TRF']['ps'],
                                        'rs_feature_auditory_word_trf':curr_results['auditory']['feature']['word']['TRF']['rs'],
                                        'ps_feature_auditory_word_trf':curr_results['auditory']['feature']['word']['TRF']['ps'],
                                        'rs_feature_visual_word_per_split_trf':curr_results['visual']['feature']['word']['TRF']['rs_per_split'],
                                        'ps_feature_visual_word_per_split_trf':curr_results['visual']['feature']['word']['TRF']['ps_per_split'],
                                        'rs_feature_auditory_word_per_split_trf':curr_results['auditory']['feature']['word']['TRF']['rs_per_split'],
                                        'ps_feature_auditory_word_per_split_trf':curr_results['auditory']['feature']['word']['TRF']['ps_per_split'],
                                        # TRF feature sentence
                                        'rs_feature_visual_sentence_trf':curr_results['visual']['feature']['sentence']['TRF']['rs'],
                                        'ps_feature_visual_sentence_trf':curr_results['visual']['feature']['sentence']['TRF']['ps'],
                                        'rs_feature_auditory_sentence_trf':curr_results['auditory']['feature']['sentence']['TRF']['rs'],
                                        'ps_feature_auditory_sentence_trf':curr_results['auditory']['feature']['sentence']['TRF']['ps'],
                                        'rs_feature_visual_sentence_per_split_trf':curr_results['visual']['feature']['sentence']['TRF']['rs_per_split'],
                                        'ps_feature_visual_sentence_per_split_trf':curr_results['visual']['feature']['sentence']['TRF']['ps_per_split'],
                                        'rs_feature_auditory_sentence_per_split_trf':curr_results['auditory']['feature']['sentence']['TRF']['rs_per_split'],
                                        'ps_feature_auditory_sentence_per_split_trf':curr_results['auditory']['feature']['sentence']['TRF']['ps_per_split'],
                                        'rf_feature_visual_sentence_per_split_trf':curr_results['visual']['feature']['sentence']['TRF']['rf_per_split'],
                                        'rf_feature_auditory_sentence_per_split_trf':curr_results['auditory']['feature']['sentence']['TRF']['rf_per_split']},
                                        # TRF full word
                                        #'rs_full_visual_word_trf':curr_results['visual']['full']['word']['TRF']['rs'],
                                        #'ps_full_visual_word_trf':curr_results['visual']['full']['word']['TRF']['ps'],
                                        #'rs_full_auditory_word_trf':curr_results['auditory']['full']['word']['TRF']['rs'],
                                        #'ps_full_auditory_word_trf':curr_results['auditory']['full']['word']['TRF']['ps'],
                                        #'rs_full_visual_word_per_split_trf':curr_results['visual']['full']['word']['TRF']['rs_per_split'],
                                        #'ps_full_visual_word_per_split_trf':curr_results['visual']['full']['word']['TRF']['ps_per_split'],
                                        #'rs_full_auditory_word_per_split_trf':curr_results['auditory']['full']['word']['TRF']['rs_per_split'],
                                        #'ps_full_auditory_word_per_split_trf':curr_results['auditory']['full']['word']['TRF']['ps_per_split'],
                                        # TRF full sentence
                                        #'rs_full_visual_sentence_trf':curr_results['visual']['full']['sentence']['TRF']['rs'],
                                        #'ps_full_visual_sentence_trf':curr_results['visual']['full']['sentence']['TRF']['ps'],
                                        #'rs_full_auditory_sentence_trf':curr_results['auditory']['full']['sentence']['TRF']['rs'],
                                        #'ps_full_auditory_sentence_trf':curr_results['auditory']['full']['sentence']['TRF']['ps'],
                                        #'rs_full_visual_sentence_per_split_trf':curr_results['visual']['full']['sentence']['TRF']['rs_per_split'],
                                        #'ps_full_visual_sentence_per_split_trf':curr_results['visual']['full']['sentence']['TRF']['ps_per_split'],
                                        #'rs_full_auditory_sentence_per_split_trf':curr_results['auditory']['full']['sentence']['TRF']['rs_per_split'],
                                        #'ps_full_auditory_sentence_per_split_trf':curr_results['auditory']['full']['sentence']['TRF']['ps_per_split'],
                                        #'rf_full_visual_sentence_per_split_trf':curr_results['visual']['full']['sentence']['TRF']['rf_per_split'],
                                        #'rf_full_auditory_sentence_per_split_trf':curr_results['auditory']['full']['sentence']['TRF']['rf_per_split'],
                                        # Evoked feature word
                                        #'rs_feature_visual_word_evoked':curr_results['visual']['feature']['word']['evoked']['rs'],
                                        #'ps_feature_visual_word_evoked':curr_results['visual']['feature']['word']['evoked']['ps'],
                                        #'rs_feature_auditory_word_evoked':curr_results['auditory']['feature']['word']['evoked']['rs'],
                                        #'ps_feature_auditory_word_evoked':curr_results['auditory']['feature']['word']['evoked']['ps'],
                                        #'rs_feature_visual_word_per_split_evoked':curr_results['visual']['feature']['word']['evoked']['rs_per_split'],
                                        #'ps_feature_visual_word_per_split_evoked':curr_results['visual']['feature']['word']['evoked']['ps_per_split'],
                                        #'rs_feature_auditory_word_per_split_evoked':curr_results['auditory']['feature']['word']['evoked']['rs_per_split'],
                                        #'ps_feature_auditory_word_per_split_evoked':curr_results['auditory']['feature']['word']['evoked']['ps_per_split'],
                                        # Evoked full word
                                        #'rs_full_visual_word_evoked':curr_results['visual']['full']['word']['evoked']['rs'],
                                        #'ps_full_visual_word_evoked':curr_results['visual']['full']['word']['evoked']['ps'],
                                        #'rs_full_auditory_word_evoked':curr_results['auditory']['full']['word']['evoked']['rs'],
                                        #'ps_full_auditory_word_evoked':curr_results['auditory']['full']['word']['evoked']['ps'],
                                        #'rs_full_visual_word_per_split_evoked':curr_results['visual']['full']['word']['evoked']['rs_per_split'],
                                        #'ps_full_visual_word_per_split_evoked':curr_results['visual']['full']['word']['evoked']['ps_per_split'],
                                        #'rs_full_auditory_word_per_split_evoked':curr_results['auditory']['full']['word']['evoked']['rs_per_split'],
                                        #'ps_full_auditory_word_per_split_evoked':curr_results['auditory']['full']['word']['evoked']['ps_per_split']},
                                        ignore_index=True)

            print(df)
            if not df.empty:
                fn = f'../../Output/encoding_models/encoding_results_case_study_{data_type}_{filt}_{feature_group}_decimate_{args.decimate}_smooth_{args.smooth}_patients_{"_".join(patients.split())}.json'
                df.to_json(fn)
                print(f'JSON file saved to: {fn}')
