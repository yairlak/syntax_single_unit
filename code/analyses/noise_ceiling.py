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
from scipy.stats import pearsonr

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(description='Train a TRF model')
# DATA
parser.add_argument('--patient', action='append', default=['505'])
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
parser.add_argument('--query-train', default="block in [1,3,5] and word_length>1",
                    help='E.g., limits to first phone in auditory blocks\
                        "and first_phone == 1"')
parser.add_argument('--query-test', default=None,
                    help='If not empry, eval model on a separate test query')
parser.add_argument('--scale-epochs', default=False, action='store_true',
                    help='If true, data is scaled *after* epoching')

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
if not args.query_test:
    args.query_test = args.query_train
print(args)

#############
# LOAD DATA #
#############
data = DataHandler(args.patient, args.data_type, args.filter,
                   args.probe_name, args.channel_name, args.channel_num)
# Both neural and feature data into a single raw object
data.load_raw_data(args.decimate)
# sfreq_original = data.raws[0].info['sfreq']  # used later for word epoch

##################
# SENTENCE-LEVEL #
##################
data.epoch_data(level='sentence_onset',
                tmin=0, tmax=1,  # Takes only first 1sec of sentence for corr
                query=args.query_train,
                smooth=args.smooth,
                scale_epochs=False,  # must be same as word level
                verbose=True)
y_sentence = data.epochs[0].copy().pick_types(seeg=True, eeg=True).\
             get_data().transpose([2, 0, 1])
metadata_sentences = data.epochs[0].metadata
sentence_strings = sorted(list(set(metadata_sentences['sentence_string'].to_list())))

##############
# WORD-LEVEL #
##############
data.epoch_data(level='word',
                tmin=0, tmax=1,
                query=args.query_train,
                scale_epochs=False,  # same for train
                verbose=False)
y_word = data.epochs[0].copy().pick_types(seeg=True, eeg=True).\
         get_data().transpose([2, 0, 1])
metadata_words = data.epochs[0].metadata

# INIT RESULTS DICT
results = {}
results['noise_ceiling'] = {}
results['noise_ceiling']['total_score'] = []  # len = num outer cv splits
results['noise_ceiling']['scores_by_time'] = []  # len = num outer cv splits
results['noise_ceiling']['rf_sentence'] = []  # len = num outer cv splits

n_sentences, n_outputs = len(sentence_strings), y_sentence.shape[2]
r_sentence = np.empty([n_sentences, 3, n_outputs])
r_sentence[:] = np.nan
for i_string, sentence_string in enumerate(sentence_strings):
    IXs_sentences = np.where(metadata_sentences['sentence_string'] == sentence_string)[0]
    assert len(IXs_sentences) == 3
    for i_sent, IX_sentence_string in enumerate(IXs_sentences):
        sentence_string = metadata_sentences['sentence_string'].iloc[IX_sentence_string]
        # neural activity of target sentence
        data_curr_sentence = y_sentence[:, i_sent, :] # n_times X n_outputs
        # neural activity of other repetitions of sentence
        IXs_other_sentences = list(set(IXs_sentences) - set([IX_sentence_string]))
        assert len(IXs_other_sentences) == 2
        data_other_sentence = y_sentence[:, IXs_other_sentences, :].mean(axis=1) # n_times X n_outputs
        for i_out in range(n_outputs):
            # Pearson correlation
            r_sentence[i_string, i_sent, i_out], _ = \
                pearsonr(data_curr_sentence[:, i_out],
                         data_other_sentence[:, i_out])

mean_r = r_sentence.mean(axis=1).mean(axis=0)
for ch_name, r in zip(data.epochs[0].ch_names, mean_r):
    print(ch_name, r)


########
# SAVE #
########
# FNAME
list_args2fname = ['patient', 'data_type', 'filter', 'smooth', 'query_train']
args2fname = args.__dict__.copy()
fname = dict2filename(args2fname, '_', list_args2fname, '', True)
print(fname)
if not os.path.exists(args.path2output):
    os.makedirs(args.path2output)
ch_names = data.epochs[0].copy().pick_types(seeg=True, eeg=True).ch_names
fn = os.path.join(args.path2output, 'noise_ceiling_' + fname + '.pkl')
with open(fn, 'wb') as f:
    pickle.dump([results, ch_names, args], f)
print(f'Results were saved to {fn}')

print(f'Run time: {datetime.datetime.now() - begin_time}')
