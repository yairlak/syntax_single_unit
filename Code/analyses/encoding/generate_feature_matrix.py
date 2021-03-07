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
from utils.data_manip import load_neural_data
from utils.features import get_features
import numpy as np
import pandas as pd
import mne

parser = argparse.ArgumentParser(description='Train an encoding model on neural data')
# DATA
parser.add_argument('--path2output', default=os.path.join('..', '..', '..', 'Output', 'encoding_models', 'features'), help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")

#############
# USER ARGS #
#############
args = parser.parse_args()
# for compatibility with load_neural_data() below:
args.patient = ['patient_479_11']
args.filter = ['gaussian-kernel']
args.probe_name = [] 
args.block_type = 'both'
args.data_type = ['micro']
args.dont_overwrite = False
print('args\n', args)

if not os.path.exists(args.path2output):
    os.makedirs(args.path2output)

#############
# LOAD DATA #
#############
args.level = 'phone'
epochs_list = load_neural_data(args)
epochs = epochs_list[0]
metadata_audio = epochs.metadata
sfreq_audio = epochs.info['sfreq']
print('Sampling frequency: ', sfreq_audio)
args.level = 'word'
epochs_list = load_neural_data(args)
epochs = epochs_list[0]
metadata_visual = epochs['block in [1, 3, 5]'].metadata
sfreq_visual = epochs.info['sfreq']
assert sfreq_audio == sfreq_visual
metadata = pd.concat([metadata_audio, metadata_visual], axis= 0)
metadata = metadata.sort_values(by='event_time')

###########################
# GENERATE FEATURE MATRIX #
###########################
feature_names = ['letters', 'word_length', 'phone_string', 'is_first_word', 'is_last_word', 'word_position', 'tense', 'pos', 'pos_simple', 'word_zipf', 'morpheme', 'morph_complex', 'grammatical_number', 'embedding', 'wh_subj_obj', 'dec_quest', 'semantic_features', 'phonological_features']
for feature_name in feature_names:
    X_features, feature_names, feature_info, feature_groups = get_features(metadata, [feature_name]) # GET DESIGN MATRIX
    print('Design matrix dimensions:', X_features.shape)
    num_samples, num_features = X_features.shape
    #print('Feature names\n', feature_names)
    print('Features\n')
    [print(k, feature_info[k]) for k in feature_info.keys()]
    
    # 
    times_sec = metadata['event_time'].values
    times_samples = (times_sec * sfreq_audio).astype(int)
    num_time_samples = int((times_sec[-1] + 10)*sfreq_audio) # Last time point plus 5sec
    X = np.zeros((num_time_samples, num_features))
    X[times_samples, :] = X_features
    
    ch_names = list(map(str, feature_names))
    ch_types = ['misc'] * len(ch_names)
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sfreq_visual)
    features_raw = mne.io.RawArray(X.T, info)
    
    
    ########
    # SAVE #
    ########
    fn = os.path.join(args.path2output, f'raw_feature_matrix_{feature_name}.fif')
    features_raw.save(fn, overwrite=True)
    print(f'Saved to: {fn}')
    
    fn = os.path.join(args.path2output, f'feature_info_{feature_name}.pkl')
    with open(fn, 'wb') as f:
        pickle.dump(feature_info, f)
    print(f'Saved to: {fn}')