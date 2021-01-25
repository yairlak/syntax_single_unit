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
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import mne
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train an encoding model on neural data')
# DATA
parser.add_argument('--patient', action='append', default=[], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], action='append', default=[], help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='word', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'gaussian-kernel-25', 'high-gamma'], action='append', default=[], help='')
parser.add_argument('--probe-name', default=[], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channe-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
# QUERY
parser.add_argument('--query', default=[], help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--feature-list', default=[], nargs='*', help='Comparison name from Code/Main/utils/comparisons.py')
parser.add_argument('--block-type', choices=['auditory', 'visual', 'both'], default='both', help='Block type will be added to the query in the comparison')
parser.add_argument('--label-from-metadata', default=[], help='Field name in metadata that will be used to generate labels for the different classes. If empty, condition_names in comparison will be used')
parser.add_argument('--pick-classes', default=[], type=str, nargs='*', help='Limit the classes to this list')
# MODEL
parser.add_argument('--model-type', default='ridge', choices=['ridge', 'lasso', 'standard']) 
parser.add_argument('--n-folds-inner', default=5, type=int, help="cross-valdition")
parser.add_argument('--n-folds-outer', default=5, type=int, help="cross-valdition")
# MISC
parser.add_argument('--tmin', default=-0.1, type=float, help='')
parser.add_argument('--tmax', default=0.7, type=float, help='')
parser.add_argument('--min-trials', default=15, type=float, help='Minimum number of trials from each class.')
parser.add_argument('--decimate', default=[], type=float, help='If not empty, (for speed) decimate data by the provided factor.')
parser.add_argument('--path2output', default=os.path.join('..', '..', '..', 'Output', 'encoding_models'), help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--dont-overwrite', default=False, action='store_true', help="If True then file will be overwritten")

#############
# USER ARGS #
#############
args = parser.parse_args()
args.patient = ['patient_' + p for p in  args.patient]
#if not args.probe_name:
#    args.probe_name = ['All']
print('args\n', args)
assert len(args.patient)==len(args.data_type)==len(args.filter)==len(args.probe_name)
# FNAME 
#list_args2fname = ['patient', 'data_type', 'filter', 'level', 'block_type', 'model_type', 'ch_name', 'feature_list', 'query']
list_args2fname = ['patient', 'data_type', 'filter', 'level', 'block_type', 'model_type', 'ch_name', 'query']

if not os.path.exists(args.path2output):
    os.makedirs(args.path2output)

np.random.seed(1)

#############
# LOAD DATA #
#############
epochs_list = load_neural_data(args)
#for epochs in epochs_list: # ADD MORE FEATURE COLUMNS TO METADATA
#    df = extend_metadata(epochs.metadata)
print(epochs_list[0])
metadata = epochs_list[0].metadata
times = epochs_list[0].times
print(len(times))
# DEBUG
#times = times[:10] # DEBUG!
print(metadata['word_string'])
#print(metadata['word_position'])
X, feature_values, feature_info, feature_groups = get_features(metadata, args.feature_list) # GET DESIGN MATRIX
print('Design matrix dimensions:', X.shape)
num_samples, num_features = X.shape
#print('Feature names\n', feature_names)
print('Features\n')
[print(k, feature_info[k]) for k in feature_info.keys()]

###############
# STANDARDIZE #
###############
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


##################
# ENCODING MODEL #
##################
for epochs in epochs_list:
    for ch_name in epochs.ch_names:
        # ADD CH_NAME TO FNAME
        args.ch_name = ch_name
        args2fname = args.__dict__.copy()
        fname = dict2filename(args2fname, '_', list_args2fname, '', True)
        
        # GET NEURAL DATA
        pick_ch = mne.pick_channels(epochs.ch_names, [ch_name])
        y = np.squeeze(epochs.get_data()[:, pick_ch, :]) # num_trials X num_timepoints
        print('Target variable dimenstions:', y.shape)
        num_timepoints = y.shape[1]
        
        ##############
        # INIT MODEL #
        ##############
        inner_cv = KFold(n_splits=args.n_folds_inner, shuffle=True, random_state=0)
        outer_cv = KFold(n_splits=args.n_folds_outer, shuffle=True, random_state=0)
        
        p_grid = {'alpha': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}
        if args.model_type == 'ridge':
            model = GridSearchCV(estimator=linear_model.Ridge(), param_grid=p_grid, cv=inner_cv)
        elif args.model_type == 'lasso':
            model = GridSearchCV(estimator=linear_model.Lasso(), param_grid=p_grid, cv=inner_cv)
        
        
        scores = {}
        # LOOP OVER FEATURE NAMES (E.G., WORD_LENGTH, WORD_ZIPF,...)
        for feature_name in args.feature_list + ['full']:
            print(f'\nComputing scores for full model or without feature: {feature_name}')
            # INIT SCORES DICT
            scores[feature_name] = {}
            scores[feature_name]['mean'] = []
            scores[feature_name]['std'] = []
        
            # REMOVE COLUMNS OF GIVEN FEATURE FROM DESIGN MATRIX
            if feature_name != 'full':
                st, ed = feature_info[feature_name]['IXs']
                X_reduced = np.delete(X, range(st,ed), 1)
            
            # LOOP OVER TIME POINTS
            for i_t, _ in enumerate(tqdm(times)):
                # SCORE
                nested_scores = cross_val_score(model, X=X_reduced, y=y[:, i_t], cv=outer_cv)  
                # APPEND                    
                scores[feature_name]['mean'].append(nested_scores)
                scores[feature_name]['std'].append(nested_scores)
            
        
        ########
        # SAVE #
        ########
        fn = os.path.join(args.path2output, fname+ '.pkl')
        with open(fn, 'wb') as f:
            pickle.dump([model, scores, args], f)
        print(f'Results were saved to {fn}')
        