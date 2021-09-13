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
from utils.utils import dict2filename
from utils.data_manip import DataHandler

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(description='Train a TRF model')
# DATA
parser.add_argument('--patient', action='append', default=['479_11'])
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike'],
                    action='append', default=['micro'], help='electrode type')
parser.add_argument('--filter', action='append', default=['raw'],
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
y_sentence = data.epochs[0].copy().pick_types(seeg=True, eeg=True).get_data().\
        transpose([2, 0, 1])
metadata_sentences = data.epochs[0].metadata

# INIT RESULTS DICT
results = {}
results['noise_ceiling'] = {}
results['noise_ceiling']['total_score'] = []  # len = num outer cv splits
results['noise_ceiling']['scores_by_time'] = []  # len = num outer cv splits
results['noise_ceiling']['rf_sentence'] = []  # len = num outer cv splits


for block_type in ['visual', 'auditory']:
    IXs_block = np.where(metadata_sentences['block_type'] == block_type)
    





########
# SAVE #
########
# FNAME
list_args2fname = ['patient', 'data_type', 'filter', 'smooth', 'model_type',
                   'probe_name', 'ablation_method',
                   'query_train', 'each_feature_value']
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
