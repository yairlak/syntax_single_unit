#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:48:59 2021

@author: yl254115
"""

import argparse, os, sys, pickle, copy, datetime
import model_manip
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
import numpy as np
from sklearn.model_selection import KFold
from sklearn import linear_model
import mne
from sklearn.preprocessing import StandardScaler
import pandas as pd
from mne.decoding import ReceptiveField
from utils.features import get_features
from utils.read_logs_and_features import extend_metadata
from utils.utils import dict2filename
from utils.data_manip import load_neural_data, get_events



parser = argparse.ArgumentParser(description='Train an encoding model on neural data')
# DATA
parser.add_argument('--patient', action='append', default=[], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], action='append', default=[], help='electrode type')
# parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='sentence_onset', help='')
parser.add_argument('--filter', action='append', default=[], help='raw/high-gamma/gaussian-kernel-xx')
parser.add_argument('--probe-name', default=[], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channe-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
parser.add_argument('--sfreq', default=1000, help='Sampling frequency for both neural and feature data (must be identical).')
# QUERY
parser.add_argument('--query', default=[], help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--scale-epochs', default=False, action='store_true', help='If true, data is scaled (StandardScalar) after epoching')
# parser.add_argument('--feature-list', default=['word_length', 'is_first_word', 'is_last_word', 'word_position', 'tense', 'pos_simple', 'word_zipf', 'morph_complex', 'grammatical_number', 'embedding', 'wh_subj_obj', 'dec_quest', 'semantic_features', 'letters'], nargs='*', help='Comparison name from Code/Main/utils/comparisons.py')
#parser.add_argument('--feature-list', default=['word_length', 'is_first_word', 'is_last_word', 'word_position', 'tense', 'pos_simple', 'word_zipf', 'morph_complex', 'grammatical_number', 'embedding', 'wh_subj_obj', 'dec_quest', 'semantic_features', 'phonological_features', 'letters'], nargs='*', help='Comparison name from Code/Main/utils/comparisons.py')
# word_length word_position tense pos_simple morph_complex embedding wh_subj_obj semantic_features
#parser.add_argument('--feature-list', default=['is_first_word', 'is_last_word', 'phonological_features', 'word_zipf', 'pos_simple', 'grammatical_number'], nargs='*', help='Comparison name from Code/Main/utils/comparisons.py')
parser.add_argument('--feature-list', default=[], nargs='*', help='Comparison name from Code/Main/utils/comparisons.py')
parser.add_argument('--label-from-metadata', default=[], help='Field name in metadata that will be used to generate labels for the different classes. If empty, condition_names in comparison will be used')
parser.add_argument('--pick-classes', default=[], type=str, nargs='*', help='Limit the classes to this list')
# MODEL
parser.add_argument('--model-type', default='ridge', choices=['ridge', 'ridge_laplacian', 'lasso', 'no-regularization']) 
parser.add_argument('--ablation-method', default='remove', choices=['shuffle', 'remove', 'zero'], help='Method used to calcuated feature importance by reducing/ablating a feature family')
parser.add_argument('--n-folds-inner', default=2, type=int, help="cross-valdition")
parser.add_argument('--n-folds-outer', default=2, type=int, help="cross-valdition")
# MISC
parser.add_argument('--tmin', default=-0.6, type=float, help='')
parser.add_argument('--tmax', default=0.8, type=float, help='')
parser.add_argument('--tmin_rf', default=-0.1, type=float, help='')
parser.add_argument('--tmax_rf', default=0.5, type=float, help='')
parser.add_argument('--decimate', default=20, type=float, help='If not empty, (for speed) decimate data by the provided factor.')
# PATHS
parser.add_argument('--path2output', default=os.path.join('..', '..', '..', 'Output', 'encoding_models'), help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--dont-overwrite', default=False, action='store_true', help="If True then file will be overwritten")

begin_time = datetime.datetime.now()
np.random.seed(1)
#############
# USER ARGS #
#############
args = parser.parse_args()
args.patient = ['patient_' + p for p in args.patient]
args.block_type = 'both'
print('args\n', args)
assert len(args.patient) == len(args.data_type) == len(args.filter)
# FNAME
list_args2fname = ['patient', 'data_type', 'filter', 'model_type',
                   'probe_name', 'ablation_method', 'query']
args2fname = args.__dict__.copy()
fname = dict2filename(args2fname, '_', list_args2fname, '', True)
print(fname)
if not os.path.exists(args.path2output):
    os.makedirs(args.path2output)

###########################
# GENERATE FEATURE MATRIX #
###########################

# GET METADATA
_, _, metadata_phone = get_events(args.patient[0], 'phone', args.data_type)
_, _, metadata_word = get_events(args.patient[0], 'word', args.data_type)

metadata_audio = extend_metadata(metadata_phone)
metadata_visual = metadata_word.query('block in [1, 3, 5]')
metadata_visual = extend_metadata(metadata_visual)

metadata_features = pd.concat([metadata_audio, metadata_visual], axis=0)
metadata_features = metadata_features.sort_values(by='event_time')

# CREATE DESIGN MATRIX
X_features, feature_names, feature_info, feature_groups = \
                        get_features(metadata_features, args.feature_list)
num_samples, num_features = X_features.shape
times_sec = metadata_features['event_time'].values
times_samples = (times_sec * args.sfreq).astype(int)
num_time_samples = int((times_sec[-1] + 10)*args.sfreq)  # add 10sec for RF
X = np.zeros((num_time_samples, num_features))
X[times_samples, :] = X_features

#################################
# STANDARIZE THE FEATURE MATRIX #
#################################
scaler = StandardScaler()
# EXCEPT FOR SENTENCE ONSET
IX_sentence_onset = feature_names.index('is_first_word')
sentence_onset = X[:, IX_sentence_onset]
X = scaler.fit_transform(X)
X[:, IX_sentence_onset] = sentence_onset

# MNE-ize feature data
ch_types = ['misc'] * len(feature_names)
info = mne.create_info(len(feature_names), ch_types=ch_types, sfreq=args.sfreq)
raw_features = mne.io.RawArray(X.T, info)
del X

####################
# LOAD NEURAL DATA #
####################
args_temp = copy.deepcopy(args)
args_temp.tmin, args_temp.tmax, args_temp.decimate = [], [], []
args_temp.level = 'sentence_onset'
epochs_neural_sentence = load_neural_data(args_temp)[0]
del args_temp

# Epoch features to sentences
epochs_features_sentence = mne.Epochs(raw_features,
                                      epochs_neural_sentence.events,
                                      tmin=epochs_neural_sentence.tmin,
                                      tmax=epochs_neural_sentence.tmax,
                                      metadata=epochs_neural_sentence.metadata,
                                      baseline=None)
epochs_features_sentence = epochs_features_sentence[args.query]

# DECIMATE
if args.decimate:
    epochs_neural_sentence = epochs_neural_sentence.decimate(args.decimate)
    epochs_features_sentence = epochs_features_sentence.decimate(args.decimate)
sfreq_down_sampled = epochs_neural_sentence.info['sfreq']

# GET SENTENCE-LEVEL DATA BEFORE SPLIT
# num_timepoints X num_epochs X num_features
X_sentence = epochs_features_sentence.get_data().transpose([2, 0, 1])
y_sentence = epochs_neural_sentence.get_data().transpose([2, 0, 1])


def reduce_design_matrix(X, feature_name, feature_info, ablation_method):
    if feature_name == 'full':
        X_reduced = X
    else:
        st, ed = feature_info[feature_name]['IXs']
        if ablation_method == 'remove':
            X_reduced = np.delete(X, range(st, ed), 2)
        elif ablation_method == 'zero':
            X_reduced = X.copy()
            X_reduced[:, :, st:ed] = 0
        elif ablation_method == 'shuffle':
            X_reduced = X.copy()
            X_reduced_FOI = X_reduced[:, :, st:ed]
            X_reduced_FOI[X_reduced_FOI != 0] = \
                np.random.permutation(X_reduced_FOI[X_reduced_FOI != 0])
            X_reduced[:, :, st:ed] = X_reduced_FOI
    return X_reduced


##################
# ENCODING MODEL #
##################
results = {}
outer_cv = KFold(n_splits=args.n_folds_outer, shuffle=True, random_state=0)

for i_split, (train, test) in enumerate(outer_cv.split(X_sentence.transpose([1,2,0]), y_sentence.transpose([1,2,0]))):
    for feature_name in ['full'] + args.feature_list: # LOOP OVER FEATURE NAMES (E.G., WORD_LENGTH, WORD_ZIPF,...)
        # INIT RESULTS DICT
        if i_split == 0:
            results[feature_name] = {}
            results[feature_name]['total_score'] = [] # list for all splits
            results[feature_name]['scores_by_time'] = [] # len of list = number of outer cv splits
            results[feature_name]['rf_sentence'] = [] # len of list = number of outer cv splits
    
        print(f'\n Split {i_split+1}/{args.n_folds_outer}, feature {feature_name}')
        # REMOVE COLUMNS OF TARGET FEATURE FROM DESIGN MATRIX
        X_sentence_reduced = reduce_design_matrix(X_sentence, feature_name, feature_info, args.ablation_method)
        
        ###############
        # TRAIN MODEL #
        ###############
        if args.ablation_method == 'remove' or (args.ablation_method in ['zero', 'shuffle'] and feature_name == 'full'):
            print(f'\nTrain TRF model: X - {X_sentence_reduced[:, train, :].shape}, y - {y_sentence[:, train, :].shape}')
            rf_sentence = model_manip.train_TRF(X_sentence_reduced[:, train, :], y_sentence[:, train, :], sfreq_down_sampled, args)
            results[feature_name]['rf_sentence'].append(rf_sentence)
        ##############
        # EVAL MODEL #
        ##############
        # SENTENCE LEVEL (SCORE FOR ENTIRE SENTENCE)
        print('Sentence level: score for the entire duration')
        results[feature_name]['total_score'].append(rf_sentence.score(X_sentence_reduced[:, test, :], y_sentence[:, test, :]))
        # WORD LEVEL
        print('Prepare test data in word level')
        sentences_test, blocks_test = epochs_neural_sentence[test].metadata['sentence_string'].tolist(), epochs_neural_sentence[test].metadata['block'].tolist()
        X_test_word, y_test_word, times_word = model_manip.get_test_data_word(raw_features, sentences_test, blocks_test, args)
        X_test_word_reduced = reduce_design_matrix(X_test_word, feature_name, feature_info, args.ablation_method)
        # TODO: Get valid samples - the following is to get valid_samples; can be avoided by calcuating it directly.
        if i_split == 0 and feature_name == 'full':
            print('Get valid samples for word data')
            estimator = linear_model.RidgeCV(alphas=[1], alpha_per_target=True)    
            rf_word = ReceptiveField(args.tmin_rf, args.tmax_rf, sfreq_down_sampled, estimator=estimator, scoring='r2', n_jobs=-1)
            rf_word.fit(X_test_word[:,:2,:], y_test_word[:,:2,:])
            valid_samples = rf_word.valid_samples_
        print('Word level: scoring per each time point')
        results[feature_name]['scores_by_time'].append(model_manip.eval_TRF_across_epochs(rf_sentence, X_test_word_reduced, y_test_word, valid_samples, args))
    del rf_sentence
results['times_word_epoch'] = times_word[valid_samples]    

########
# SAVE #
########
ch_names = epochs_neural_sentence.ch_names
fn = os.path.join(args.path2output, fname+ '.pkl')
with open(fn, 'wb') as f:
    pickle.dump([results, ch_names, args, feature_info], f)
print(f'Results were saved to {fn}')

print(f'Run time: {datetime.datetime.now() - begin_time}')
