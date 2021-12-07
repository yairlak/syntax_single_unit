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
from encoding.model_manip import reduce_design_matrix
from utils.utils import dict2filename
from utils.data_manip import DataHandler

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy import stats

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
parser.add_argument('--smooth', default=50, type=int,
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
parser.add_argument('--decimate', default=50, type=int,
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
data.epoch_data(level='word',
                query=args.query_train,
                smooth=args.smooth,
                scale_epochs=False,  # must be same as word level
                verbose=True)

# PREPARE MATRICES
X = data.epochs[0].copy().pick_types(misc=True).get_data()

# TAKE FEATURES AT TIME ZERO (FEATURE VALUES SHOULD BE ZERO ELSEWHERE)
times = data.epochs[0].times
IXs = np.where(times==0)
X = X[:, :, IXs[0][0]]
X = np.expand_dims(X, axis=2)
y = data.epochs[0].copy().pick_types(seeg=True, eeg=True).get_data()
n_epochs, n_channels, n_times = y.shape
metadata = data.epochs[0].metadata

# FOR COMPATABILITY BELOW, ADD SINGLETON FOR TIME DIMENSION
#X = np.repeat(X[:, :, np.newaxis], n_times, axis=2)

##################
# ENCODING MODEL #
##################
feature_names = ['full']  # first append full model, which is mandatory

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
    results[feature_name]['total_score'] = []  #
    results[feature_name]['scores_by_time'] = []  # Same
    results[feature_name]['stats_by_time'] = []  # Same
    results[feature_name]['rf_sentence'] = []  # Same

# DEFINE MODEL
alphas = np.logspace(-3, 8, 100) 
estimator = RidgeCV(alphas=alphas,
                    alpha_per_target = True,
                    cv=None, # efficient LOO
                    scoring='r2')
kf_out = KFold(n_splits=args.n_folds_outer,
               shuffle=True,
               random_state=1)
for feature_name in feature_names:
    print(f'\n feature: {feature_name}')
    # REMOVE COLUMNS OF TARGET FEATURE FROM DESIGN MATRIX
    print(f'X shape: {X.shape}')
    X_reduced = reduce_design_matrix(X.transpose([2, 0, 1]),
                                     feature_name,
                                     data.feature_info,
                                     args.ablation_method)
    # TRANSPOSE/RESHAPE
    X_reduced = X_reduced.transpose([1, 2, 0]) # back to standard MNE order
    X_reduced = X_reduced[:, :, 0] # remove singleton at time dimension (axis=2)
    y = y.reshape([n_epochs, -1])
    print(f'\nTrain model: X (n_trials, n_features) - {X_reduced.shape}, \
          y (n_trials, n_times * n_outputs) - {y.shape}')
    
    y_pred_cv, y_true_cv = [], []
    for i_split, (IXs_train, IXs_test) in enumerate(kf_out.split(X_reduced, y)):
        print(f'Split {i_split}')
        estimator.fit(X_reduced[IXs_train, :], y[IXs_train, :])
        y_pred = estimator.predict(X_reduced[IXs_test, :])
        y_pred_cv.append(y_pred)
        y_true_cv.append(y[IXs_test, :])   
    
    y_pred = np.vstack(y_pred_cv)
    y_true = np.vstack(y_true_cv)

    rs, ps = [],[]
    for i in range(y_pred.shape[1]): # n_times * n_outputs
        r, p = stats.spearmanr(y_pred[:, i],
                               y_true[:, i])   
        rs.append(r)
        ps.append(p)
    scores_by_time = np.asarray(rs).reshape([n_channels, n_times])
    stats_by_time = np.asarray(ps).reshape([n_channels, n_times])
    results[feature_name]['total_score'].append(None)
    results[feature_name]['scores_by_time'].append(scores_by_time)
    results[feature_name]['stats_by_time'].append(stats_by_time)
    print(f'\nWord-level test score: maximal r = {scores_by_time.max():.3f}')
        
results['times_word_epoch'] = None

########
# SAVE #
########
# FNAME
list_args2fname = ['patient', 'data_type', 'filter', 'decimate', 'smooth', 'model_type',
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
fn = os.path.join(args.path2output, 'evoked_' + fname + '.pkl')
with open(fn, 'wb') as f:
    pickle.dump([results, ch_names, args, data.feature_info], f)
print(f'Results were saved to {fn}')

print(f'Run time: {datetime.datetime.now() - begin_time}')
