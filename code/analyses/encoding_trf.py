#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:48:59 2021

@author: yl254115
"""
import argparse
import os
import pickle
import datetime
import numpy as np
from sklearn.model_selection import KFold
from encoding.model_manip import reduce_design_matrix,\
                                 eval_TRF_across_epochs, train_TRF
from utils.utils import dict2filename
from utils.data_manip import DataHandler
from sklearn.preprocessing import StandardScaler

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(description='Train a TRF model')
# DATA
parser.add_argument('--patient', action='append', default=[])
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike'],
                    action='append', default=[], help='electrode type')
parser.add_argument('--filter', action='append', default=[],
                    help='raw/high-gamma')
parser.add_argument('--smooth', default=25, type=int,
                    help='Gaussian-kernal width in milisec or None')
parser.add_argument('--probe-name', default=None, nargs='*',
                    action='append', type=str,
                    help='Probe name to plot (ignores args.channel-name/num)')
parser.add_argument('--channel-name', default=None, nargs='*', action='append',
                    type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=None, nargs='*', action='append',
                    type=int, help='If empty list then all channels are taken')
parser.add_argument('--sfreq', default=1000,
                    help='Sampling frequency for both neural and feature data \
                    (must be identical).')
# QUERY
parser.add_argument('--query-train', default="block in [2,4,6] and word_length>1",
                    help='E.g., limits to first phone in auditory blocks\
                        "and first_phone == 1"')
parser.add_argument('--query-test', default=None,
                    help='If not empry, eval model on a separate test query')
parser.add_argument('--scale-epochs', default=False, action='store_true',
                    help='If true, data is scaled *after* epoching')
# FEATURES
#parser.add_argument('--feature-list',
#                    default=['is_first_word',
#                              'is_last_word',
#                              'phonological_features'],
#                    nargs='*',
#                    help='Feature to include in the encoding model')
parser.add_argument('--feature-list',
                    nargs='*',
#                    action='append',
                    default=None,
                    help='Feature to include in the encoding model')
parser.add_argument('--each-feature-value', default=False, action='store_true',
                    help="Evaluate model after ablating each feature value. \
                         If false, ablate all feature values together")
# MODEL
parser.add_argument('--model-type', default='ridge',
                    choices=['ridge', 'ridge_laplacian', 'lasso'])
parser.add_argument('--ablation-method', default='remove',
                    choices=['zero', 'remove', 'shuffle'],
                    help='Method to use for calcuating feature importance')
parser.add_argument('--n-folds-inner', default=5, type=int, help="For CV")
parser.add_argument('--n-folds-outer', default=5, type=int, help="For CV")
parser.add_argument('--train-only', default=False, action='store_true',
                    help="Train model and save, without model evaluation")
parser.add_argument('--eval-only', default=False, action='store_true',
                    help="Evaluate model without training \
                        (requires pre-trained models)")
# MISC
parser.add_argument('--tmin_word', default=-1.2, type=float,
                    help='Start time of word time window')
parser.add_argument('--tmax_word', default=1, type=float,
                    help='End time of word time window')
parser.add_argument('--tmin_rf', default=0, type=float,
                    help='Start time of receptive-field kernel')
parser.add_argument('--tmax_rf', default=1, type=float,
                    help='End time of receptive-field kernel')
parser.add_argument('--decimate', default=20, type=float,
                    help='Set empty list for no decimation.')
# PATHS
parser.add_argument('--path2output',
                    default=os.path.join
                    ('..', '..', 'Output', 'encoding_models'),
                    help="Path to where trained models and results are saved")

begin_time = datetime.datetime.now()
np.random.seed(1)
#############
# USER ARGS #
#############
args = parser.parse_args()
assert len(args.patient) == len(args.data_type) == len(args.filter)
args.patient = ['patient_' + p for p in args.patient]
args.block_type = 'both'
if not args.query_test:
    args.query_test = args.query_train
if isinstance(args.feature_list, str):
    args.feature_list = eval(args.feature_list)
print(args)

#############
# LOAD DATA #
#############
data = DataHandler(args.patient, args.data_type, args.filter,
                   args.probe_name, args.channel_name, args.channel_num,
                   args.feature_list)
# Both neural and feature data into a single raw object
data.load_raw_data(args.decimate)
# sfreq_original = data.raws[0].info['sfreq']  # used later for word epoch
# GET SENTENCE-LEVEL DATA BEFORE SPLIT
data.epoch_data(level='sentence_onset',
                query=args.query_train,
                smooth=args.smooth,
                scale_epochs=False,  # must be same as word level
                verbose=True)

# print(set(data.epochs[0].metadata['sentence_string']))
# PREPARE MATRICES
X_sentence = data.epochs[0].copy().pick_types(misc=True).get_data().\
        transpose([2, 0, 1])

y_sentence = data.epochs[0].copy().pick_types(seeg=True, eeg=True).get_data().\
        transpose([2, 0, 1])

metadata_sentences = data.epochs[0].metadata


##################
# ENCODING MODEL #
##################
feature_names = ['full']  # performance of the full model is mandatory

# ABLATE FEATURE VALUES (TOGETHER OR SEPARATELY)
if args.each_feature_value:
    for f in data.feature_info.keys():
        for f_name in data.feature_info[f]['names']:
            feature_names.append(f_name)
else:
    feature_names.extend(args.feature_list)
# INIT RESULTS DICT
results = {}
for feature_name in feature_names:
    results[feature_name] = {}
    results[feature_name]['total_score'] = []  # len = num outer cv splits
    results[feature_name]['scores_by_time'] = []  # len = num outer cv splits
    results[feature_name]['rf_sentence'] = []  # len = num outer cv splits

outer_cv = KFold(n_splits=args.n_folds_outer, shuffle=True, random_state=0)
for i_split, (train, test) in enumerate(outer_cv.split(
                                        X_sentence.transpose([1, 2, 0]),
                                        y_sentence.transpose([1, 2, 0]))):
    
    # WORD LEVEL
    print('Prepare test data at word level')
    sentences_test = metadata_sentences['sentence_string'].to_numpy()[test]
    blocks_test = metadata_sentences['block'].to_numpy()[test]
    
    query_test_sentences = ' or '.join(
        [f'(sentence_string=="{s}" and block=={b})'
         for s, b in zip(sentences_test, blocks_test)])
    data.epoch_data(level='word',
                    tmin=args.tmin_word, tmax=args.tmax_word,
                    query=f'({args.query_test}) and \
                        ({query_test_sentences})',
                    scale_epochs=False,  # same for train
                    verbose=False)
    # print(set(data.epochs[0].metadata['word_string']))
    X_test_word = data.epochs[0].copy().pick_types(misc=True).get_data().\
        transpose([2, 0, 1])
    
    y_test_word = data.epochs[0].copy().pick_types(seeg=True, eeg=True).\
        get_data().transpose([2, 0, 1])
    
    for feature_name in feature_names:
        print(f'\n Split {i_split+1}/{args.n_folds_outer}, {feature_name}')
        # REMOVE COLUMNS OF TARGET FEATURE FROM DESIGN MATRIX
        X_sentence_reduced = reduce_design_matrix(X_sentence, feature_name,
                                                  data.feature_info,
                                                  args.ablation_method)
        ###############
        # TRAIN MODEL #
        ###############
        if args.ablation_method == 'remove' or \
            (args.ablation_method in ['zero', 'shuffle'] and
             feature_name == 'full'):
            print(f'\nTrain TRF: X (n_times, n_trials, n_features)- \
                  {X_sentence_reduced[:, train, :].shape}, \
                  y (n_times, n_trials, n_outputs) - {y_sentence[:, train, :].shape}')
            rf_sentence = train_TRF(
                                    X_sentence_reduced[:, train, :],
                                    y_sentence[:, train, :],
                                    data.sfreq, args)
            results[feature_name]['rf_sentence'].append(rf_sentence)

        ##############
        # EVAL MODEL #
        ##############

        # SENTENCE LEVEL (SCORE FOR ENTIRE SENTENCE)
        print('Sentence level: score for the entire duration')
        score_sentence = rf_sentence.score(X_sentence_reduced[:, test, :],
                                           y_sentence[:, test, :])
        print(f'Sentence-level test score: r = {score_sentence[0]:.3f}')
        results[feature_name]['total_score'].append(score_sentence)

        # WORD LEVEL (SCORE PER TIME POINT)
        if args.ablation_method in ['zero', 'shuffle']:
            start_sample = int(data.sfreq * args.tmin_word)
        else:
            start_sample = 0
        X_test_word_reduced = reduce_design_matrix(X_test_word, feature_name,
                                                   data.feature_info,
                                                   args.ablation_method,
                                                   start_sample)
        delays = np.arange(int(np.round(args.tmin_rf * data.sfreq)),
                           int(np.round(args.tmax_rf * data.sfreq) + 1))
        min_delay = None if delays[-1] <= 0 else delays[-1]
        max_delay = None if delays[0] >= 0 else delays[0]
        valid_samples = slice(min_delay, max_delay)
        print('Word level: scoring per each time point')
        scores_by_time = eval_TRF_across_epochs(rf_sentence,
                                                X_test_word_reduced,
                                                y_test_word,
                                                valid_samples,
                                                args)
        results[feature_name]['scores_by_time'].append(scores_by_time)
        print(f'\nWord-level test score: maximal r = {scores_by_time.max():.3f}')
        
results['times_word_epoch'] = data.epochs[0].times[valid_samples]

########
# SAVE #
########
# FNAME
list_args2fname = ['patient', 'data_type', 'filter',
                   'decimate', 'smooth', 'model_type',
                   'probe_name', 'ablation_method',
                   'query_train', 'feature_list', 'each_feature_value']
if args.query_train != args.query_test:
    list_args2fname.extend(['query_test'])
args2fname = args.__dict__.copy()
fname = dict2filename(args2fname, '_', list_args2fname, '', True)
print(fname)
if not os.path.exists(args.path2output):
    os.makedirs(args.path2output)
ch_names = data.epochs[0].copy().pick_types(seeg=True, eeg=True).ch_names
fn = os.path.join(args.path2output, fname + '.pkl')
with open(fn, 'wb') as f:
    pickle.dump([results, ch_names, args, data.feature_info], f)
print(f'Results were saved to {fn}')

print(f'Run time: {datetime.datetime.now() - begin_time}')
