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
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
import mne
from sklearn.preprocessing import StandardScaler
from mne.decoding import ReceptiveField, TimeDelayingRidge
import pandas as pd
from utils.features import get_features

parser = argparse.ArgumentParser(description='Train an encoding model on neural data')
# DATA
parser.add_argument('--patient', action='append', default=[], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], action='append', default=[], help='electrode type')
# parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='sentence_onset', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'gaussian-kernel-25', 'high-gamma'], action='append', default=[], help='')
parser.add_argument('--probe-name', default=[], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channe-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
# QUERY
parser.add_argument('--query', default=[], help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
# parser.add_argument('--feature-list', default=['word_length', 'is_first_word', 'is_last_word', 'word_position', 'tense', 'pos_simple', 'word_zipf', 'morph_complex', 'grammatical_number', 'embedding', 'wh_subj_obj', 'dec_quest', 'semantic_features', 'letters'], nargs='*', help='Comparison name from Code/Main/utils/comparisons.py')
# word_length word_position tense pos_simple morph_complex embedding wh_subj_obj semantic_features
parser.add_argument('--feature-list', default=[], nargs='*', help='Comparison name from Code/Main/utils/comparisons.py')
parser.add_argument('--label-from-metadata', default=[], help='Field name in metadata that will be used to generate labels for the different classes. If empty, condition_names in comparison will be used')
parser.add_argument('--pick-classes', default=[], type=str, nargs='*', help='Limit the classes to this list')
# MODEL
parser.add_argument('--model-type', default='ridge', choices=['ridge', 'ridge_laplacian', 'lasso', 'standard']) 
parser.add_argument('--n-folds-inner', default=5, type=int, help="cross-valdition")
parser.add_argument('--n-folds-outer', default=5, type=int, help="cross-valdition")
# MISC
parser.add_argument('--tmin', default=-0.6, type=float, help='')
parser.add_argument('--tmax', default=0.8, type=float, help='')
parser.add_argument('--tmin_rf', default=-0.1, type=float, help='')
parser.add_argument('--tmax_rf', default=0.7, type=float, help='')
parser.add_argument('--decimate', default=0, type=float, help='If not empty, (for speed) decimate data by the provided factor.')
# PATHS
parser.add_argument('--path2output', default=os.path.join('..', '..', '..', 'Output', 'encoding_models'), help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--dont-overwrite', default=False, action='store_true', help="If True then file will be overwritten")

np.random.seed(1)
#############
# USER ARGS #
#############
args = parser.parse_args()
args.patient = ['patient_' + p for p in  args.patient]
args.block_type = 'both' # allow only querying
print('args\n', args)
assert len(args.patient)==len(args.data_type)==len(args.filter)
# FNAME
list_args2fname = ['patient', 'data_type', 'filter', 'model_type', 'query', 'probe_name']
args2fname = args.__dict__.copy()
fname = dict2filename(args2fname, '_', list_args2fname, '', True)
print(fname)
if not os.path.exists(args.path2output):
    os.makedirs(args.path2output)

####################
# LOAD NEURAL DATA #
####################
tmin, tmax, decimate = args.tmin, args.tmax, args.decimate
args.tmin, args.tmax, args.decimate = [], [], []
args.level = 'phone'
epochs_phone = load_neural_data(args)[0]
args.level = 'sentence_onset'
epochs_sentence = load_neural_data(args)[0]
args.tmin, args.tmax  = tmin, tmax
args.level = 'word'
epochs_word = load_neural_data(args)[0]
ch_names = epochs_sentence.ch_names
assert epochs_sentence.info['sfreq'] == epochs_word.info['sfreq'] == epochs_phone.info['sfreq']
sfreq_features = epochs_word.info['sfreq']

metadata_audio = epochs_phone.metadata
metadata_visual = epochs_word['block in [1, 3, 5]'].metadata
metadata_features = pd.concat([metadata_audio, metadata_visual], axis= 0)
metadata_features = metadata_features.sort_values(by='event_time')

args.decimate = decimate

###########################
# GENERATE FEATURE MATRIX #
###########################
X_features, feature_names, feature_info, feature_groups = get_features(metadata_features, args.feature_list) # GET DESIGN MATRIX
num_samples, num_features = X_features.shape
times_sec = metadata_features['event_time'].values
times_samples = (times_sec * sfreq_features).astype(int)
num_time_samples = int((times_sec[-1] + 10)*sfreq_features) # Last time point plus 5sec
X = np.zeros((num_time_samples, num_features))
X[times_samples, :] = X_features


# STANDARDIZE 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MNE-ize
# create info and raw
ch_types = ['misc'] * len(feature_names)
info = mne.create_info(len(feature_names), ch_types=ch_types, sfreq=sfreq_features)
raw_features = mne.io.RawArray(X_scaled.T, info)
# epoch features to sentences
epochs_features_sentence = mne.Epochs(raw_features, epochs_sentence.events, tmin=epochs_sentence.tmin, tmax=epochs_sentence.tmax, metadata=epochs_sentence.metadata, baseline=None)
epochs_features_sentence = epochs_features_sentence[args.query]
# epoch features to words
epochs_features_word = mne.Epochs(raw_features, epochs_word.events, tmin=epochs_word.tmin, tmax=epochs_word.tmax, metadata=epochs_word.metadata, baseline=None)
epochs_features_word = epochs_features_word[args.query]
# decimate
if args.decimate:
    epochs_sentence = epochs_sentence.decimate(args.decimate)
    epochs_word = epochs_word.decimate(args.decimate)
    epochs_features_sentence = epochs_features_sentence.decimate(args.decimate)
    epochs_features_word = epochs_features_word.decimate(args.decimate)


##################
# ENCODING MODEL #
##################
results = {}    
for feature_name in ['full'] + args.feature_list: # LOOP OVER FEATURE NAMES (E.G., WORD_LENGTH, WORD_ZIPF,...)
    results[feature_name] = {}
    results[feature_name]['total_score'] = {}
    results[feature_name]['scores_by_time'] = {}
    results[feature_name]['rf_sentence'] = {}
    i_split = 0
    results[feature_name]['scores_by_time'][f'split-{i_split}'] = []
    
    print(f'\nComputing scores for full model without feature: {feature_name}')
    
    #outer_cv = KFold(n_splits=args.n_folds_outer, shuffle=True, random_state=0)
    # for i_split in range(3):
    
    ############
    # GET DATA #
    ############
    X_sentence = epochs_features_sentence.get_data().transpose([2,0,1])
    X_word = epochs_features_word.get_data().transpose([2,0,1])
    y_sentence = epochs_sentence.get_data() # num_trials X num_channels X num_timepoints
    y_sentence = np.transpose(y_sentence, [2, 0, 1]) # num_timepoints X num_channels X num_trials
    y_word = epochs_word.get_data()
    y_word = np.transpose(y_word, [2, 0, 1])

    
    ##############
    # INIT MODEL #
    ##############
    alphas = np.logspace(-3, 10, 100)
    # REMOVE COLUMNS OF GIVEN FEATURE FROM DESIGN MATRIX
    if feature_name == 'full':
        X_sentence_reduced = X_sentence.copy()
        X_word_reduced = X_word.copy()
    else:
        st, ed = feature_info[feature_name]['IXs']
        X_sentence_reduced = np.delete(X_sentence, range(st,ed), 2)
        X_word_reduced = np.delete(X_word, range(st,ed), 2)
    
    
    if args.model_type == 'ridge':
        estimator = linear_model.RidgeCV(alphas=alphas, alpha_per_target=True)
    elif args.model_type == 'lasso':
        estimator = linear_model.LassoCV()
    elif args.model_type == 'ridge_laplacian':
        estimator = TimeDelayingRidge(args.tmin, args.tmax, epochs_sentence.info['sfreq'], reg_type=['laplacian', 'ridge'], alpha=alphas, n_jobs=-2)

    ###############
    # TRAIN MODEL #
    ###############
    rf_sentence = ReceptiveField(args.tmin_rf, args.tmax_rf, epochs_sentence.info['sfreq'], 
                #feature_names=feature_values, 
                estimator=estimator, scoring='r2', n_jobs=-2)

    rf_word = ReceptiveField(args.tmin_rf, args.tmax_rf, epochs_word.info['sfreq'], 
                #feature_names=feature_values, 
                estimator=estimator, scoring='r2', n_jobs=-2)
    
    #########
    # SCORE #
    #########    
    print(f'Split {i_split+1}/{args.n_folds_outer}')
    print('Input variable dimenstions:', X_sentence_reduced.shape)
    print('Target variable dimenstions:', y_sentence.shape)
    rf_sentence.fit(X_sentence_reduced, y_sentence)
    rf_word.fit(X_word_reduced[:,:2,:], y_word[:,:2,:])
    y_pred = rf_sentence.predict(X_word_reduced)
    y_pred = y_pred[rf_word.valid_samples_]
    y_masked = y_word[rf_word.valid_samples_]
    
    n_times, n_epochs, n_outputs = y_masked.shape
    for t in range(n_times):
        curr_score = r2_score(y_masked[t,:,:], y_pred[t,:,:], multioutput='raw_values')
        results[feature_name]['scores_by_time'][f'split-{i_split}'].append(curr_score)
    
    results[feature_name]['scores_by_time'][f'split-{i_split}'] = np.asarray(results[feature_name]['scores_by_time'][f'split-{i_split}'])
    results[feature_name]['total_score'][f'split-{i_split}'] = rf_sentence.score(X_sentence_reduced, y_sentence)
    results[feature_name]['rf_sentence'][f'split-{i_split}'] = rf_sentence
    print(results[feature_name]) 

results['times_word_epoch'] = epochs_word.times[rf_word.valid_samples_]

########
# SAVE #
########
fn = os.path.join(args.path2output, fname+ '.pkl')
with open(fn, 'wb') as f:
    pickle.dump([results, ch_names, args, feature_info], f)
print(f'Results were saved to {fn}')

