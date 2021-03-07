#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:48:59 2021

@author: yl254115
"""

import argparse, os, sys, pickle
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
from utils.utils import dict2filename
from utils.data_manip import load_neural_data
from utils.features import get_features
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train an encoding model on neural data')
# DATA
parser.add_argument('--patient', action='append', default=['479_11'], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], action='append', default=['micro'], help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='sentence_onset', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'gaussian-kernel-25', 'high-gamma'], action='append', default=['gaussian-kernel'], help='')
parser.add_argument('--probe-name', default=[['LSTG']], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channe-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
# QUERY
parser.add_argument('--query', default=['block in [2,4,6]'], help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--feature-list', default=['word_length', 'is_first_word', 'is_last_word', 'word_position', 'tense', 'pos_simple', 'word_zipf', 'morph_complex', 'grammatical_number', 'embedding', 'wh_subj_obj', 'dec_quest', 'semantic_features', 'phonological_features'], nargs='*', help='Comparison name from Code/Main/utils/comparisons.py')
parser.add_argument('--label-from-metadata', default=[], help='Field name in metadata that will be used to generate labels for the different classes. If empty, condition_names in comparison will be used')
parser.add_argument('--pick-classes', default=[], type=str, nargs='*', help='Limit the classes to this list')
# MODEL
parser.add_argument('--model-type', default='ridge', choices=['ridge', 'ridge_laplacian', 'lasso', 'standard']) 
parser.add_argument('--n-folds-inner', default=5, type=int, help="cross-valdition")
parser.add_argument('--n-folds-outer', default=5, type=int, help="cross-valdition")
# MISC
parser.add_argument('--tmin', default=-0.1, type=float, help='')
parser.add_argument('--tmax', default=0.7, type=float, help='')
parser.add_argument('--min-trials', default=15, type=float, help='Minimum number of trials from each class.')
parser.add_argument('--decimate', default=40, type=float, help='If not empty, (for speed) decimate data by the provided factor.')
# PATHS
parser.add_argument('--path2output', default=os.path.join('..', '..', '..', 'Output', 'encoding_models'), help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--dont-overwrite', default=False, action='store_true', help="If True then file will be overwritten")

#############
# USER ARGS #
#############
args = parser.parse_args()
args.patient = ['patient_' + p for p in  args.patient]
args.block_type = 'both' # allow only querying
#if not args.probe_name:
#    args.probe_name = ['All']
print('args\n', args)
assert len(args.patient)==len(args.data_type)==len(args.filter)==len(args.probe_name)
# FNAME 
#list_args2fname = ['patient', 'data_type', 'filter', 'level', 'block_type', 'model_type', 'ch_name', 'feature_list', 'query']
list_args2fname = ['patient', 'data_type', 'filter', 'level', 'model_type', 'query', 'probe_name']

if not os.path.exists(args.path2output):
    os.makedirs(args.path2output)

np.random.seed(1)



####################
# LOAD NEURAL DATA #
####################
tmin, tmax, decimate = args.tmin, args.tmax, args.decimate
args.tmin, args.tmax, args.decimate = [], [], []
epochs_list = load_neural_data(args)
args.tmin, args.tmax, args.decimate = tmin, tmax, decimate

epochs = epochs_list[0]
epochs.decimate(args.decimate)
num_epochs, num_channels, num_times = epochs.get_data().shape
print(epochs)
metadata = epochs.metadata
times = epochs.times
print(len(times))
sfreq = epochs.info['sfreq']
print('Sampling frequency: ', sfreq)
print(metadata['word_string'])


#################
# LOAD FEATURES #
#################

X, feature_values_names = np.empty((num_epochs, 0, num_times)),[]
feature_info_all = {}
for feature_name in args.feature_list:
    # Load feature matrix
    raw_features = mne.io.read_raw_fif(os.path.join(args.path2output, 'features', f'raw_feature_matrix_{feature_name}.fif'), verbose=False)
    epochs_features = mne.Epochs(raw_features, epochs.events, tmin=epochs.tmin, tmax=epochs.tmax, metadata=epochs.metadata)
    epochs_features.decimate(args.decimate)
    epochs_features = epochs_features[args.query]
    feature_info = pickle.load(open(os.path.join(args.path2output, 'features', f'feature_info_{feature_name}.pkl'), 'rb'))
    st = X.shape[1]
    X = np.concatenate((X, epochs_features.get_data()), axis=1) 
    ed = X.shape[1]
    feature_info[feature_name]['IXs'] = (st, ed) 
    feature_info_all = {**feature_info_all, **feature_info}
    feature_values_names.extend(feature_info[feature_name]['names'])
X = X.transpose([2,0,1])
    
###############
# STANDARDIZE #
###############
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X) # num_epochs X num_features
print(X.shape)
##################
# ENCODING MODEL #
##################
for epochs in epochs_list:
    # ADD CH_NAME TO FNAME
    # args.ch_name = ch_name
    args2fname = args.__dict__.copy()
    fname = dict2filename(args2fname, '_', list_args2fname, '', True)
    print(fname)
    
    # GET NEURAL DATA
    y = epochs.get_data() # num_trials X num_channels X num_timepoints
    y = np.transpose(y, [2, 0, 1]) # num_timepoints X num_channels X num_trials
    print('Target variable dimenstions:', y.shape)
    num_timepoints = y.shape[1]
    
    ##############
    # INIT MODEL #
    ##############
    
    alphas = np.logspace(-3, 10, 100)
    
    
    scores, rfs = {}, []
    # LOOP OVER FEATURE NAMES (E.G., WORD_LENGTH, WORD_ZIPF,...)
    for feature_name in ['full'] + args.feature_list:
        print(f'\nComputing scores for full model without feature: {feature_name}')
        # INIT SCORES DICT
        scores[feature_name] = []
        
        # REMOVE COLUMNS OF GIVEN FEATURE FROM DESIGN MATRIX
        if feature_name == 'full':
            X_reduced = X.copy()
        else:
            st, ed = feature_info_all[feature_name]['IXs']
            X_reduced = np.delete(X, range(st,ed), 2)
        
        
        if args.model_type == 'ridge':
            estimator = linear_model.RidgeCV(alphas=alphas, alpha_per_target=True)
        elif args.model_type == 'lasso':
            estimator = linear_model.LassoCV()
        elif args.model_type == 'ridge_laplacian':
            estimator = TimeDelayingRidge(args.tmin, args.tmax, sfreq, reg_type=['laplacian', 'ridge'], alpha=alphas, n_jobs=-2)

        rf = ReceptiveField(args.tmin, args.tmax, sfreq, 
                    #feature_names=feature_values, 
                    estimator=estimator, scoring='corrcoef', n_jobs=-2)

        
    
        # rfs = []
        # outer_cv = KFold(n_splits=args.n_folds_outer, shuffle=True, random_state=0)
        # for i_split,(train, test) in enumerate(outer_cv.split(X_reduced, y)):
        #     print(f'Split {i_split+1}/{args.n_folds_outer}')
        X_reduced[np.isnan(X_reduced)]=0
        rf.fit(X_reduced, y)
        # rfs.append(rf)
        scores[feature_name].append(rf.score(X_reduced, y))
        print(rf.score(X_reduced, y)) 
        rfs.append(rf)
    
    ########
    # SAVE #
    ########
    fn = os.path.join(args.path2output, fname+ '.pkl')
    with open(fn, 'wb') as f:
        pickle.dump([rfs, scores, args, feature_info_all], f)
    print(f'Results were saved to {fn}')
    
