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
parser.add_argument('--feature-list',
                    nargs='*',
#                    action='append',
                    default=None,
                    help='Feature to include in the encoding model')
#parser.add_argument('--query-train', default="block in [2,4,6] and word_length>1")
#parser.add_argument('--query-test', default="block in [2,4,6] and word_length>1")
parser.add_argument('--each-feature-value', default=False, action='store_true',
                    help="Evaluate model after ablating each feature value. \
                         If false, ablate all feature values together")


#############
# USER ARGS #
#############

args = parser.parse_args()
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

df = pd.DataFrame()

dict_hemi = {'L':'Left', 'R':'Right'}

print('Collecting results...')


patients="479_11 479_25 482 499 502 505 510 513 515 530 538 539 540 541 543 544 549 551"
patients="479_11"
data_types = ['micro', 'macro', 'spike']
data_types = ['micro']
filters = ['raw', 'high-gamma']
filters = ['raw']

args.probe_name = None
for patient in patients.split():
    args.patient = ['patient_' + patient]
    for data_type in data_types:
        args.data_type = data_type
        for filt in filters:
            if data_type=='spike' and filt != 'raw':
                continue
            args.filter = filt
            results_evoked = {'auditory':None, 'visual':None}
            results_trf = {'auditory':None, 'visual':None}
            feature_info_evoked = {'auditory':None, 'visual':None}
            feature_info_trf = {'auditory':None, 'visual':None}
            found_both_blocks = True
            for block in ['auditory', 'visual']:
                print(f'Patient - {patient}, {data_type}, {filt}, {block}') 

                args.query_train = {'auditory':'block in [2,4,6] and word_length>1',
                                    'visual':'block in [1,3,5] and word_length>1'}[block]
                args.feature_list = {'auditory':"position_phonology_lexicon_semantics_syntax".split('_'),
                                     'visual':"position_orthography_lexicon_semantics_syntax".split('_')}[block]
                list_args2fname = ['patient', 'data_type', 'filter',
                                   'decimate', 'smooth', 'model_type',
                                   'probe_name', 'ablation_method',
                                   'query_train', 'feature_list', 'each_feature_value']

                args2fname = args.__dict__.copy()

                # EVOKED
                fname = 'evoked_' + dict2filename(args2fname, '_', list_args2fname, '', True)
                try:
                    results_evoked[block], ch_names_evoked, args_evoked, feature_info_evoked[block] = \
                        pickle.load(open(os.path.join(args.path2output, fname + '.pkl'), 'rb'))
                except:
                    print(f'File not found: {args.path2output}/{fname}.pkl')
                    found_both_blocks = False
                    continue
                
                # TRF
                fname = dict2filename(args2fname, '_', list_args2fname, '', True)
                try:
                    results_trf[block], ch_names_trf, args_trf, feature_info_trf[block] = \
                        pickle.load(open(os.path.join(args.path2output, fname + '.pkl'), 'rb'))
                except:
                    print(f'File not found: {args.path2output}/{fname}.pkl')
                    found_both_blocks = False
                    continue
            
            if not found_both_blocks:
                print(f'Skipping patient {patient}, {data_type}, {filt}')
                continue

            assert ch_names_evoked==ch_names_trf            
            for i_ch, ch_name in enumerate(ch_names_trf):
                probe_name = get_probe_name(ch_name, args.data_type)
                feature_info_keys = list(feature_info_trf['auditory'].keys()) + list(feature_info_trf['auditory'].keys())
                for feature in list(set(feature_info_keys)) + ['full']:
                    
                    # TRF
                    scores_by_time_trf, stats_by_time_trf = {}, {}
                    for block in ['auditory', 'visual']:
                        scores_by_time_trf[block] = {}
                        stats_by_time_trf[block] = {}
                        if feature in results_trf[block].keys():
                            # FEATURE (scores by time)
                            scores_by_time[block]['feature'] = results_trf[block][feature]['scores_by_time']
                            stats_by_time[block]['feature'] = results_trf[block][feature]['stats_by_time']
                            # FULL MODEL (LATER USED FOR CALCULATING DELTA-R)
                            # FULL MODEL (BY-TIME)
                            scores_by_time[block]['full'] = results_trf[block]['full']['scores_by_time'][i_ch, :]
                            stats_by_time[block]['full'] = results_trf[block]['full']['stats_by_time'][i_ch, :]
                        else:
                            scores_by_time[block]['feature'] = None
                            stats_by_time[block]['feature'] = None
                            scores_by_time[block]['full'] = None
                            stats_by_time[block]['full'] = None
                    
                    
                    if args.data_type == 'spike':
                        st = 5
                    else:
                        st = 0
                    if probe_name[0] in dict_hemi.keys():
                        hemi = dict_hemi[probe_name[st]]
                    else:
                        hemi = None


                    df = df.append({'data_type':data_type,
                                    'filter':filt,
                                    'Probe_name':probe_name,
                                    'Hemisphere':hemi,
                                    'Ch_name':ch_name,
                                    'Patient':patient, 
                                    'Feature':feature,
                                    'r_visual_by_time_trf':scores_by_time['visual']['feature'],
                                    'r_auditory_by_time_trf':scores_by_time['auditory']['feature'],
                                    'r_full_visual_by_time_trf':scores_by_time['visual']['full'],
                                    'r_full_auditory_by_time_trf':scores_by_time['auditory']['full'],
                                    'stats_visual_by_time_trf':stats_by_time['visual']['feature'],
                                    'stats_auditory_by_time_trf':stats_by_time['auditory']['feature'],
                                    'stats_full_visual_by_time_trf':stats_by_time['visual']['full'],
                                    'stats_full_auditory_by_time_trf':stats_by_time['auditory']['full'],
                                    'r_visual_by_time_evoked':scores_by_time['visual']['feature'],
                                    'r_auditory_by_time_evoked':scores_by_time['auditory']['feature'],
                                    'r_full_visual_by_time_evoked':scores_by_time['visual']['full'],
                                    'r_full_auditory_by_time_evoked':scores_by_time['auditory']['full'],
                                    'stats_visual_by_time_evoked':stats_by_time['visual']['feature'],
                                    'stats_auditory_by_time_evoked':stats_by_time['auditory']['feature'],
                                    'stats_full_visual_by_time_evoked':stats_by_time['visual']['full'],
                                    'stats_full_auditory_by_time_evoked':stats_by_time['auditory']['full']},
                                    ignore_index=True)

print(df)
if not df.empty:
    fn = f'../../Output/encoding_models/evoked_encoding_results_decimate_{args.decimate}_smooth_{args.smooth}.json'
    df.to_json(fn)
    print(f'JSON file saved to: {fn}')
