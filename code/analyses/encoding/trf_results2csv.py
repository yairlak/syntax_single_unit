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
from viz import plot_rf_coefs, plot_rf_r2
sys.path.append('..')
from utils.utils import dict2filename
import matplotlib.pyplot as plt
from encoding.models import TimeDelayingRidgeCV
import numpy as np
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser(description='Plot TRF results')
# DATA
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike'],
                    action='append', default=[], help='electrode type')
parser.add_argument('--filter', action='append',
                    default=[],
                    help='raw/high-gamma')
parser.add_argument('--smooth', default=50,
                    help='Gaussian smoothing in msec')
parser.add_argument('--probe-name', default=None, nargs='*',
                    action='append', type=str,
                    help='Probe name to plot (ignores channel-name/num)')
parser.add_argument('--channel-name', default=[], nargs='*', action='append',
                    type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=[], nargs='*', action='append',
                    type=int, help='channel number (if empty all channels)')
# MISC
parser.add_argument('--path2output',
                    default=os.path.join('..', '..', '..',
                                         'Output', 'encoding_models'))
parser.add_argument('--path2figures',
                    default=os.path.join('..', '..', '..',
                                         'Figures', 'encoding_models', 'scatters'))
parser.add_argument('--decimate', default=50.0, type=float,
                    help='If not empty, decimate data for speed.')
parser.add_argument('--model-type', default='ridge',
                    choices=['ridge', 'lasso', 'ridge_laplacian', 'standard'])
parser.add_argument('--ablation-method', default='remove',
                    choices=['shuffle', 'remove', 'zero'],
                    help='Method used to calcuated feature importance\
                        by reducing/ablating a feature family')
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
    return probe_name

df = pd.DataFrame()

dict_hemi = {'L':'Left', 'R':'Right'}

print('Collecting results...')

patients="479_11 479_25 482 502 505 510 513 515 530 538 539 540 541 543"


for patient in patients.split():
    print(f'Patient - {patient}') 
    results = {'auditory':None, 'visual':None}
    found_both_blocks = True
    for block in ['auditory', 'visual']:

        args.query_train = {'auditory':'block in [2,4,6] and word_length>1',
                            'visual':'block in [1,3,5] and word_length>1'}[block]
        args.feature_list = {'auditory':['is_first_word', 'word_onset'] + "positional_phonology_lexicon_syntax".split('_'),
                             'visual':['is_first_word', 'word_onset'] + "positional_orthography_lexicon_syntax".split('_')}[block]
        args.patient = ['patient_' + patient]
        list_args2fname = ['patient', 'data_type', 'filter',
                           'decimate', 'smooth', 'model_type',
                           'probe_name', 'ablation_method',
                           'query_train', 'feature_list', 'each_feature_value']


        args2fname = args.__dict__.copy()
        fname = dict2filename(args2fname, '_', list_args2fname, '', True)

        try:
            results[block], ch_names, args_trf, feature_info = \
                pickle.load(open(os.path.join(args.path2output, fname + '.pkl'), 'rb'))
        except:
            print(f'File not found: {args.path2output}/{fname}.pkl')
            found_both_blocks = False
            continue
    
    if not found_both_blocks:
        print(f'Skipping patient {patient}')
        continue

    for i_ch, ch_name in enumerate(ch_names):
        probe_name = get_probe_name(ch_name, args.data_type[0])
        for feature in list(set(list(feature_info.keys()) + ['orthography', 'phonology'])) + ['full']:
            
            dict_cv_score, mean_score = {}, {}
            for block in ['auditory', 'visual']:
                dict_cv_score[block] = {}
                mean_score[block] = {}
                dict_cv_score[block]['feature'] = {}
                dict_cv_score[block]['full'] = {}
                mean_score[block]['feature'] = {}
                mean_score[block]['full'] = {}
                if feature in results[block].keys():
                    # FEATURE
                    total_score_all_CVs_channels = results[block][feature]['total_score']
                    cv_score = []
                    for cv in range(len(total_score_all_CVs_channels)):
                        cv_score.append(total_score_all_CVs_channels[cv][i_ch])
                    dict_cv_score[block]['feature']['total'] = cv_score
                    mean_score[block]['feature']['total'] = np.mean(dict_cv_score[block]['feature']['total'])
                    
                    # FEATURE (scores by time)
                    scores_by_time_all_CVs_channels = results[block][feature]['scores_by_time']
                    cv_score = []
                    for cv in range(len(scores_by_time_all_CVs_channels)):
                        #print(len(scores_by_time_all_CVs_channels))
                        #print(scores_by_time_all_CVs_channels[0].shape)
                        cv_score.append(scores_by_time_all_CVs_channels[cv][:, i_ch])
                    cv_score = np.asarray(cv_score)
                    dict_cv_score[block]['feature']['by_time'] = cv_score
                    mean_score[block]['feature']['by_time'] = cv_score.mean(axis=0)
                    
                    # FULL MODEL (LATER USED FOR CALCULATING DELTA-R)
                    total_score_all_CVs_channels = results[block]['full']['total_score']
                    cv_score = []
                    for cv in range(len(total_score_all_CVs_channels)):
                        cv_score.append(total_score_all_CVs_channels[cv][i_ch])
                    dict_cv_score[block]['full']['total'] = cv_score
                    mean_score[block]['full']['total'] = np.mean(dict_cv_score[block]['full']['total'])
                    
                    # FULL MODEL (BY-TIME)
                    scores_by_time_all_CVs_channels = results[block]['full']['scores_by_time']
                    cv_score = []
                    for cv in range(len(scores_by_time_all_CVs_channels)):
                        cv_score.append(scores_by_time_all_CVs_channels[cv][:, i_ch])
                    cv_score = np.asarray(cv_score)
                    dict_cv_score[block]['full']['by_time'] = cv_score
                    mean_score[block]['full']['by_time'] = cv_score.mean(axis=0) 
                else:
                    dict_cv_score[block]['feature']['total'] = None
                    dict_cv_score[block]['feature']['by_time'] = None
                    mean_score[block]['feature']['total'] = None
                    mean_score[block]['feature']['by_time'] = None
                    dict_cv_score[block]['full']['total'] = None
                    dict_cv_score[block]['full']['by_time'] = None
                    mean_score[block]['full']['total'] = None
                    mean_score[block]['full']['by_time'] = None
            

            if args.data_type == 'spike':
                st = 5
            else:
                st = 0
            if probe_name[0] in dict_hemi.keys():
                hemi = dict_hemi[probe_name[st]]
            else:
                hemi = None
            df = df.append({'Probe_name':probe_name,
                            'Hemisphere':hemi,
                            'Ch_name':ch_name,
                            'Patient':patient, 
                            'Feature':feature,
                            #'Block':block,
                            'r_visual':mean_score['visual']['feature']['total'],
                            'r_auditory':mean_score['auditory']['feature']['total'],
                            'r_visual_by_time':mean_score['visual']['feature']['by_time'],
                            'r_auditory_by_time':mean_score['auditory']['feature']['by_time'],
                            'r_full_visual':mean_score['visual']['full']['total'],
                            'r_full_auditory':mean_score['auditory']['full']['total'],
                            'r_full_visual_by_time':mean_score['visual']['full']['by_time'],
                            'r_full_auditory_by_time':mean_score['auditory']['full']['by_time'],
                            'r_CV_visual':dict_cv_score['visual']['feature']['total'],
                            'r_CV_auditory':dict_cv_score['auditory']['feature']['total']}, ignore_index=True)

print(df)
if not df.empty:
    df.to_csv(f'../../../Output/encoding_models/trf_results_{args.data_type[0]}_{args.filter[0]}.csv')
    print(f'CSV file saved to: ../../../Output/encoding_models/trf_results_{args.data_type[0]}_{args.filter[0]}.csv')
