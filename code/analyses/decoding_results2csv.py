#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:48:59 2021

@author: yl254115
"""
import argparse
import os, copy
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sys.path.append('..')
from utils import utils
from utils.utils import dict2filename, get_all_patient_numbers
import pandas as pd
from mne.stats import permutation_cluster_1samp_test
from decoding.utils import get_args2fname

parser = argparse.ArgumentParser(description='')
# DATA
parser.add_argument('--patient', action='append', default=[], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], action='append', default=[], help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default=None, help='')
parser.add_argument('--filter', choices=['raw', 'high-gamma'], action='append', default=[], help='')
parser.add_argument('--probe-name', default=[], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
parser.add_argument('--ROIs', default=None, nargs='*', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--data-type_filters', choices=['micro_high-gamma','macro_high-gamma', 'micro_raw','macro_raw', 'spike_raw'], nargs='*', default=[], help='Only if args.ROIs is used')
parser.add_argument('--smooth', default=None, type=int, help='If not empty, (for speed) decimate data by the provided factor.')
parser.add_argument('--decimate', default=None, type=int, help='If not empty, (for speed) decimate data by the provided factor.')
# QUERY
parser.add_argument('--comparison-name', default=None, help='Comparison name from Code/Main/functions/comparisons.py')
parser.add_argument('--comparison-name-test', default=None, help='Comparison name from Code/Main/functions/comparisons.py')
#parser.add_argument('--block-train', choices=['auditory', 'visual'], default='visual', help='Block type will be added to the query in the comparison')
#parser.add_argument('--block-test', choices=['auditory', 'visual'], default=None, help='Block type will be added to the query in the comparison')
parser.add_argument('--fixed-constraint', default=None, help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--fixed-constraint-test', default=None, help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--min-trials', default=10, type=float, help='Minimum number of trials from each class.')
# DECODER
parser.add_argument('--classifier', default='ridge', choices=['svc', 'logistic', 'ridge'], help='Specify a classifier type to be used')
parser.add_argument('--gat', default=False, action='store_true', help='If True, GAT will be computed; else, diagonal only')
# MISC
parser.add_argument('--tmin', default=None, type=float, help='crop window. If empty, only crops 0.1s from both sides, due to edge effects.')
parser.add_argument('--tmax', default=None, type=float, help='crop window')
parser.add_argument('--vmin', default=None, type=float, help='')
parser.add_argument('--vmax', default=None, type=float, help='')
parser.add_argument('--cat-k-timepoints', type=int, default=1, help='How many time points to concatenate before classification')
parser.add_argument('--path2figures', default='../../Figures/Decoding')
parser.add_argument('--path2output', default='../../Output/decoding')
parser.add_argument('--dont-overwrite', default=False, action='store_true', help="If True then file will be overwritten")
# PARSE
args = parser.parse_args()

if not args.ROIs:
    ROIs = list(utils.ROI2probenames().keys())
else:
    ROIs = args.ROIs.copy()

if not args.patient:
    args.patient = get_all_patient_numbers(path2data='../../Data/UCLA')
args.patient = ['patient_' + p for p in  args.patient]


if args.comparison_name == args.comparison_name_test: args.comparison_name_test = None

keys = ['fvals', 'clusters', 'cluster_p_values', 'H0', 'max_decoding', 'scores', 'data-type_filters',
        'block_train', 'block_test', 'ROI', 'comparison_name']
dict_decoding_results = {}
for k in keys:
    dict_decoding_results[k] = []

for comparison_name in ['dec_quest_len2', 'embedding_vs_long', 'number']:
    args.comparison_name = comparison_name
    for block_train in ['visual', 'auditory']:
        for block_test in ['visual', 'auditory']:
            # LOOP OVER ROIs
            for ROI in ROIs:
                # FILENAME FOR RESULTS
                args.block_train = block_train
                args.block_test = block_test
                args.ROIs = ROI
                args2fname = get_args2fname(args) # List of args
                fname_pkl = dict2filename(args.__dict__, '_', args2fname, 'pkl', True)
                fname_pkl = os.path.join(args.path2output, fname_pkl)
                try:
                    results = pickle.load(open(fname_pkl, 'rb'))
                    print(f'Reading: {fname_pkl}')
                    scores, times, tmin, tmax, time_gen, clf, args_decoding = results
                    fvals, clusters, cluster_p_values, H0 = \
                    permutation_cluster_1samp_test(np.asarray(scores)-0.5,
                                       n_permutations=1000,
                                       threshold=None, tail=0,
                                       n_jobs=-1, verbose=False,
                                       seed=42, out_type='mask')
                    dict_decoding_results['fvals'].append(fvals)
                    dict_decoding_results['clusters'].append(clusters)
                    dict_decoding_results['cluster_p_values'].append(cluster_p_values)
                    dict_decoding_results['H0'].append(H0)
                    dict_decoding_results['max_decoding'].append(scores.mean(axis=0).max())
                    dict_decoding_results['scores'].append(scores)
                    #dict_decoding_results['scores_std'] = [scores.std(axis=0)]
                    dict_decoding_results['block_train'].append(block_train)
                    dict_decoding_results['block_test'].append(block_test)
                    dict_decoding_results['ROI'].append(ROI)
                    dict_decoding_results['comparison_name'].append(args.comparison_name)
                    dict_decoding_results['data-type_filters'].append(args.data_type_filters)
                except:
                    print(f'Decoding results not found: {fname_pkl}')

df = pd.DataFrame.from_records(dict_decoding_results)
print(df)
if not df.empty:
    dtype_filters = '-'.join(args.data_type_filters)
    fn_csv = f'../../Output/decoding/decoding_results_{dtype_filters}.csv'
    df.to_csv(fn_csv, index=False)
    print(f'CSV file saved to: {fn_csv}')


#for patient in patients.split():
#    print(f'Patient - {patient}') 
#    results = {'auditory':None, 'visual':None}
#    found_both_blocks = True
#    for block in ['auditory', 'visual']:
#
#        args.query_train = {'auditory':'block in [2,4,6] and word_length>1',
#                            'visual':'block in [1,3,5] and word_length>1'}[block]
#        args.feature_list = {'auditory':['is_first_word', 'word_onset'] + "positional_phonology_lexicon_syntax".split('_'),
#                             'visual':['is_first_word', 'word_onset'] + "positional_orthography_lexicon_syntax".split('_')}[block]
#        args.patient = ['patient_' + patient]
#        list_args2fname = ['patient', 'data_type', 'filter',
#                           'decimate', 'smooth', 'model_type',
#                           'probe_name', 'ablation_method',
#                           'query_train', 'feature_list', 'each_feature_value']
#
#
#        args2fname = args.__dict__.copy()
#        fname = dict2filename(args2fname, '_', list_args2fname, '', True)
#
#        try:
#            results[block], ch_names, args_trf, feature_info = \
#                pickle.load(open(os.path.join(args.path2output, fname + '.pkl'), 'rb'))
#        except:
#            print(f'File not found: {args.path2output}/{fname}.pkl')
#            found_both_blocks = False
#            continue
#    
#    if not found_both_blocks:
#        print(f'Skipping patient {patient}')
#        continue
#
#    for i_ch, ch_name in enumerate(ch_names):
#        probe_name = get_probe_name(ch_name, args.data_type[0])
#        for feature in list(set(list(feature_info.keys()) + ['orthography', 'phonology'])) + ['full']:
#            
#            dict_cv_score, mean_score = {}, {}
#            for block in ['auditory', 'visual']:
#                dict_cv_score[block] = {}
#                mean_score[block] = {}
#                dict_cv_score[block]['feature'] = {}
#                dict_cv_score[block]['full'] = {}
#                mean_score[block]['feature'] = {}
#                mean_score[block]['full'] = {}
#                if feature in results[block].keys():
#                    # FEATURE
#                    total_score_all_CVs_channels = results[block][feature]['total_score']
#                    cv_score = []
#                    for cv in range(len(total_score_all_CVs_channels)):
#                        cv_score.append(total_score_all_CVs_channels[cv][i_ch])
#                    dict_cv_score[block]['feature']['total'] = cv_score
#                    mean_score[block]['feature']['total'] = np.mean(dict_cv_score[block]['feature']['total'])
#                    
#                    # FEATURE (scores by time)
#                    scores_by_time_all_CVs_channels = results[block][feature]['scores_by_time']
#                    cv_score = []
#                    for cv in range(len(scores_by_time_all_CVs_channels)):
#                        #print(len(scores_by_time_all_CVs_channels))
#                        #print(scores_by_time_all_CVs_channels[0].shape)
#                        cv_score.append(scores_by_time_all_CVs_channels[cv][:, i_ch])
#                    cv_score = np.asarray(cv_score)
#                    dict_cv_score[block]['feature']['by_time'] = cv_score
#                    mean_score[block]['feature']['by_time'] = cv_score.mean(axis=0)
#                    
#                    # FULL MODEL (LATER USED FOR CALCULATING DELTA-R)
#                    total_score_all_CVs_channels = results[block]['full']['total_score']
#                    cv_score = []
#                    for cv in range(len(total_score_all_CVs_channels)):
#                        cv_score.append(total_score_all_CVs_channels[cv][i_ch])
#                    dict_cv_score[block]['full']['total'] = cv_score
#                    mean_score[block]['full']['total'] = np.mean(dict_cv_score[block]['full']['total'])
#                    
#                    # FULL MODEL (BY-TIME)
#                    scores_by_time_all_CVs_channels = results[block]['full']['scores_by_time']
#                    cv_score = []
#                    for cv in range(len(scores_by_time_all_CVs_channels)):
#                        cv_score.append(scores_by_time_all_CVs_channels[cv][:, i_ch])
#                    cv_score = np.asarray(cv_score)
#                    dict_cv_score[block]['full']['by_time'] = cv_score
#                    mean_score[block]['full']['by_time'] = cv_score.mean(axis=0) 
#                else:
#                    dict_cv_score[block]['feature']['total'] = None
#                    dict_cv_score[block]['feature']['by_time'] = None
#                    mean_score[block]['feature']['total'] = None
#                    mean_score[block]['feature']['by_time'] = None
#                    dict_cv_score[block]['full']['total'] = None
#                    dict_cv_score[block]['full']['by_time'] = None
#                    mean_score[block]['full']['total'] = None
#                    mean_score[block]['full']['by_time'] = None
#            
#
#            if args.data_type == 'spike':
#                st = 5
#            else:
#                st = 0
#            if probe_name[0] in dict_hemi.keys():
#                hemi = dict_hemi[probe_name[st]]
#            else:
#                hemi = None
#            df = df.append({'Probe_name':probe_name,
#                            'Hemisphere':hemi,
#                            'Ch_name':ch_name,
#                            'Patient':patient, 
#                            'Feature':feature,
#                            #'Block':block,
#                            'r_visual':mean_score['visual']['feature']['total'],
#                            'r_auditory':mean_score['auditory']['feature']['total'],
#                            'r_visual_by_time':mean_score['visual']['feature']['by_time'],
#                            'r_auditory_by_time':mean_score['auditory']['feature']['by_time'],
#                            'r_full_visual':mean_score['visual']['full']['total'],
#                            'r_full_auditory':mean_score['auditory']['full']['total'],
#                            'r_full_visual_by_time':mean_score['visual']['full']['by_time'],
#                            'r_full_auditory_by_time':mean_score['auditory']['full']['by_time'],
#                            'r_CV_visual':dict_cv_score['visual']['feature']['total'],
#                            'r_CV_auditory':dict_cv_score['auditory']['feature']['total']}, ignore_index=True)
#
