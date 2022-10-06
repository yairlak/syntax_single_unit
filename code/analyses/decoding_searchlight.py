#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:19:32 2022

@author: yair
"""

import argparse
import os
import pickle
import pandas as pd
<<<<<<< HEAD

=======
from MNI_coords import UtilsCoords
>>>>>>> 0402d6c821bb152bb80f3e58dd8137e5009891ff
from utils.data_manip import DataHandler
from decoding.utils import update_args, get_comparisons, get_args2fname
from utils.utils import update_queries, dict2filename
from decoding.decoder import decode_comparison

<<<<<<< HEAD
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser()
# DATA
parser.add_argument('--patient', action='append', default=[],
                    help='Patient number')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'],
                    action='append', default=[])
parser.add_argument('--level',
                    choices=['sentence_onset','sentence_offset',
                             'word', 'phone'],
                    default=None)
parser.add_argument('--filter', choices=['raw', 'high-gamma'],
                    action='append', default=[], help='')
parser.add_argument('--probe-name', default=[], nargs='*', action='append',
                    type=str, help='e.g., LSTG, overrides channel_name/num')
parser.add_argument('--ROIs', default=None, nargs='*', type=str,
                    help='e.g., Brodmann.22-lh, overrides probe_name')
parser.add_argument('--channel-name', default=[], nargs='*', action='append',
                    type=str, help='e.g., GA1-LAH1')
parser.add_argument('--channel-num', default=[], nargs='*', action='append',
                    type=int, help='e.g., 3 to pick the third channel')
parser.add_argument('--responsive-channels-only', action='store_true',
                    default=False, help='Based on aud and vis files in Epochs folder')
parser.add_argument('--data-type_filters',
                    choices=['micro_high-gamma','macro_high-gamma',
                             'micro_raw','macro_raw', 'spike_raw'], nargs='*',
                             default=[], help='Only if args.ROIs is used')
parser.add_argument('--smooth', default=None, type=int,
                    help='gaussian width in [msec]')
=======

parser = argparse.ArgumentParser()
parser.add_argument('--smooth', default=50, type=int,
                    help='Gaussian-kernal width in milisec or None')
>>>>>>> 0402d6c821bb152bb80f3e58dd8137e5009891ff
parser.add_argument('--decimate', default=50, type=int)
parser.add_argument('--sfreq', default=1000,
                    help='Sampling frequency for both neural and feature data \
                    (must be identical).')
# CUBE
parser.add_argument('--x', default=24, type=float, help='X coordinate')
parser.add_argument('--y', default=23, type=float, help='Y coordinate')
parser.add_argument('--z', default=20, type=float, help='Z coordinate')
parser.add_argument('--side', default=8, type=float, help='Side of cube in mm')

# QUERY
parser.add_argument('--comparison-name', default='dec_quest_len2',
                    help='See Utils/comparisons.py')
parser.add_argument('--comparison-name-test', default=None,
                    help='See Utils/comparisons.py')
parser.add_argument('--block-train', choices=['auditory', 'visual'],
                    default='auditory',
                    help='Block type is added to the query in the comparison')
parser.add_argument('--block-test', choices=['auditory', 'visual'],
                    default=None,
                    help='Block type is added to the query in the comparison')
parser.add_argument('--fixed-constraint', default=None,
                    help='e.g., "and first_phone == 1"')
parser.add_argument('--fixed-constraint-test', default=None,
                    help='e.g., "and first_phone == 1"')
parser.add_argument('--min-trials', default=40, type=float,
                    help='Minimum number of trials from each class.')
parser.add_argument('--scale-epochs', default=False, action='store_true',
                    help='If true, data is scaled *after* epoching')
# DECODER
parser.add_argument('--classifier', default='logistic',
                    choices=['svc', 'logistic', 'ridge'])
parser.add_argument('--equalize-classes', default='downsample',
                    choices=['upsample', 'downsample'])
parser.add_argument('--gat', default=False, action='store_true',
                    help='If True, GAT will be computed; else, diagonal only')
# MISC
parser.add_argument('--tmin', default=None, type=float)
parser.add_argument('--tmax', default=None, type=float)
parser.add_argument('--cat-k-timepoints', type=int, default=1,
                    help='How many time points to concatenate before classification')
parser.add_argument('--path2output', default='../../Output/decoding')
args = parser.parse_args()

<<<<<<< HEAD
args.ROIs = None
args = update_args(args)
    
data = DataHandler(args.patient, args.data_type, args.filter,
                   None, args.channel_name, None)
data.load_raw_data(args.decimate)
data.epoch_data(level=args.level,
                query=None,
                smooth=args.smooth,
                scale_epochs=False,  # must be same as word level
                verbose=True)

# GET COMPARISONS (CONTRASTS)
comparisons = get_comparisons(args.comparison_name, # List with two dicts for
                              args.comparison_name_test) # comparison train and test

print('\nARGUMENTS:')
# pprint(args.__dict__, width=1)
if 'level' in comparisons[0].keys():
    args.level = comparisons[0]['level']
if len(comparisons[0]['queries'])>2:
    args.multi_class = True
else:
    args.multi_class = False

metadata = data.epochs[0].metadata
comparisons[0] = update_queries(comparisons[0], args.block_train, # TRAIN
                                args.fixed_constraint, metadata)
comparisons[1] = update_queries(comparisons[1], args.block_test, # TEST
                                args.fixed_constraint_test, metadata)

scores, pvals, temp_estimator, clf, stimuli, stimuli_gen = \
                    decode_comparison(data.epochs, comparisons, args)

# SAVE
args.x = round(args.x, 2)
args.y = round(args.y, 2)
args.z = round(args.z, 2)
args2fname = ['comparison_name', 'comparison_name_test',
              'block_train', 'block_test',
              'smooth', 'decimate',
              'side', 'x', 'y', 'z']   # List of args
fname_pkl = dict2filename(args.__dict__, '_', args2fname, 'pkl', True)
fname_pkl = os.path.join(args.path2output, fname_pkl)
with open(fname_pkl, 'wb') as f:
    pickle.dump([scores, pvals, data.epochs[0].times,
                 temp_estimator, clf, comparisons,
                 (stimuli, stimuli_gen), args], f)
print(f'Results saved to: {fname_pkl}')

    
=======
# LOAD COORDINATES
path2coords = '../../Data/UCLA/MNI_coords/'
fn_coords = 'electrode_locations.csv'
df = pd.read_csv(os.path.join(path2coords, fn_coords))

# PICK CHANNELS IN CUBE
df_cube = UtilsCoords.pick_channels_by_cube(df,
                                            (args.x, args.y, args.z),
                                            side=args.side,
                                            isMacro=True,
                                            isMicro=False,
                                            isStim=False)
# GET DATA AND DECODE
if not df_cube.empty:
    print(df_cube)
    dict_args = {}
    patients = df_cube['patient'].astype('str').to_list()
    data_types = df_cube['ch_type'].to_list()
    filters = ['raw'] * len(data_types)
    #channel_names = [f'{probe_name}-{ch_num}' for probe_name, ch_num in zip(df_cube['probe_name'], df_cube['ch_num'])]
    channel_names = [[e] for e in df_cube['electrode'].to_list()]
    args.patient = patients
    args.data_type = list(set(data_types))
    args.ROIs = None
    args = update_args(args)
    
    data = DataHandler(args.patient, data_types, filters,
                       None, channel_names, None)
    data.load_raw_data(args.decimate)
    data.epoch_data(level='sentence_onset',
                    query=None,
                    smooth=args.smooth,
                    scale_epochs=False,  # must be same as word level
                    verbose=True)
    
    # GET COMPARISONS (CONTRASTS)
    comparisons = get_comparisons(args.comparison_name, # List with two dicts for
                                  args.comparison_name_test) # comparison train and test

    print('\nARGUMENTS:')
    # pprint(args.__dict__, width=1)
    if 'level' in comparisons[0].keys():
        args.level = comparisons[0]['level']
    if len(comparisons[0]['queries'])>2:
        args.multi_class = True
    else:
        args.multi_class = False

    metadata = data.epochs[0].metadata
    comparisons[0] = update_queries(comparisons[0], args.block_train, # TRAIN
                                    args.fixed_constraint, metadata)
    comparisons[1] = update_queries(comparisons[1], args.block_test, # TEST
                                    args.fixed_constraint_test, metadata)

    scores, pvals, temp_estimator, clf, stimuli, stimuli_gen = \
                        decode_comparison(data.epochs, comparisons, args)
    
    # SAVE
    args.x = round(args.x, 2)
    args.y = round(args.y, 2)
    args.z = round(args.z, 2)
    args2fname = ['comparison_name', 'comparison_name_test',
                  'block_train', 'block_test',
                  'smooth', 'decimate',
                  'side', 'x', 'y', 'z']   # List of args
    fname_pkl = dict2filename(args.__dict__, '_', args2fname, 'pkl', True)
    fname_pkl = os.path.join(args.path2output, fname_pkl)
    with open(fname_pkl, 'wb') as f:
        pickle.dump([scores, pvals, data.epochs[0].times,
                     temp_estimator, clf, comparisons,
                     (stimuli, stimuli_gen), args], f)
    print(f'Results saved to: {fname_pkl}')

    
>>>>>>> 0402d6c821bb152bb80f3e58dd8137e5009891ff
