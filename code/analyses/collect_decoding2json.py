#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:48:59 2021

@author: yl254115
"""
import argparse
import os, copy
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sys.path.append('..')
from utils import utils
from utils.utils import dict2filename, get_all_patient_numbers
import pandas as pd
from mne.stats import permutation_cluster_1samp_test
from decoding.utils import get_args2fname
from mne.stats import fdr_correction

parser = argparse.ArgumentParser(description='')
# DATA
parser.add_argument('--smooth', default=None, type=int, help='If not empty, (for speed) decimate data by the provided factor.')
parser.add_argument('--decimate', default=None, type=int, help='If not empty, (for speed) decimate data by the provided factor.')
parser.add_argument('--fixed-constraint', default=None, help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--fixed-constraint-test', default=None, help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--min-trials', default=10, type=float, help='Minimum number of trials from each class.')
# DECODER
parser.add_argument('--classifier', default='ridge', choices=['svc', 'logistic', 'ridge'], help='Specify a classifier type to be used')
parser.add_argument('--gat', default=False, action='store_true', help='If True, GAT will be computed; else, diagonal only')
# MISC
parser.add_argument('--cat-k-timepoints', type=int, default=1, help='How many time points to concatenate before classification')
parser.add_argument('--path2figures', default='../../Figures/Decoding')
parser.add_argument('--path2output', default='../../Output/decoding')
parser.add_argument('--dont-overwrite', default=False, action='store_true', help="If True then file will be overwritten")
# PARSE
args = parser.parse_args()

data_types = ['micro', 'spike']
filters = ['raw', 'high-gamma']
comparison_list = ['dec_quest_len2', 'embedding_vs_long', 'number', 'word_string_first', 'pos_simple']
block_types = ['visual', 'auditory']
ROIs = list(utils.ROI2probenames().keys())

keys = ['data_type', 'data-type_filters', 'filter', 'ROI', 'probe_name',
        'block_train', 'block_test',
        'comparison_name', 'comparison_name_test', 'stimuli',
        'times', 'scores', 'pvals', 'chance_level',
        'time_gen', 'clf', 'args_decoding']

dict_decoding_results = {}
for k in keys:
    dict_decoding_results[k] = []

for data_type in data_types:
    args.data_type = data_type
    for filt in filters:
        args.filter = filt
        args.data_type_filters = f'{data_type}_{filt}'
        for comparison_name in comparison_list:
            print(comparison_name)
            args.comparison_name = comparison_name
            for block_train in block_types:
                for block_test in block_types:
                    if block_train != block_test:
                        args.comparison_name_test = comparison_name
                    else:
                        args.comparison_name_test = None
                    # LOOP OVER ROIs
                    for ROI in ROIs:
                        # FILENAME FOR RESULTS
                        args.block_train = block_train
                        if block_test != block_train:
                            args.block_test = block_test
                        else:
                            args.block_test = None
                        args.ROIs = ROI
                        args2fname = get_args2fname(args) # List of args
                        fname_pkl = dict2filename(args.__dict__, '_', args2fname, 'pkl', True)
                        fname_path2pkl = os.path.join(args.path2output, fname_pkl)
                        try:
                            results = pickle.load(open(fname_path2pkl, 'rb'))
                            print(f'Reading: {fname_pkl}')
                            scores, pvals, U1s, times, time_gen, clf, comparisons, stimuli, args_decoding = results
                            if len(pvals[0])==2: # binary case
                                pvals = [t[0] for t in pvals] # first and second sublists should be identical

                            dict_decoding_results['data_type'].append(data_type)
                            dict_decoding_results['filter'].append(filt)
                            dict_decoding_results['data-type_filters'].append(f'{data_type}_{filt}')
                            dict_decoding_results['ROI'].append(ROI)
                            dict_decoding_results['probe_name'].append(None)
                            
                            dict_decoding_results['block_train'].append(block_train)
                            dict_decoding_results['block_test'].append(block_test)
                            
                            dict_decoding_results['comparison_name'].append(args.comparison_name)
                            dict_decoding_results['comparison_name_test'].append(args.comparison_name_test)
                            dict_decoding_results['stimuli'].append(stimuli)
                            
                            dict_decoding_results['times'].append(times)
                            dict_decoding_results['scores'].append(scores)
                            dict_decoding_results['pvals'].append(pvals)
                            
                            dict_decoding_results['time_gen'].append(time_gen)
                            dict_decoding_results['clf'].append(clf)
                            dict_decoding_results['args_decoding'].append(args_decoding.__dict__)
                            chance_level = 1/len(stimuli[0])
                            dict_decoding_results['chance_level'].append(chance_level)
                        except:
                            print(f'Decoding results not found: {fname_path2pkl}')
                        scores, pvals, times, time_gen, clf, comparisons, stimuli, args_decoding = \
                            [], [], [], [], [], [], [], []

[print(k, len(dict_decoding_results[k])) for k in dict_decoding_results.keys()]
df = pd.DataFrame.from_records(dict_decoding_results)

###############
# ADD COLUMNS #
###############
def get_reject_fdr(row, alpha=0.05):
    reject_fdr, _ = fdr_correction(row['pvals'],
                                   alpha=alpha,
                                   method='indep')
    return reject_fdr

def get_pval_fdr(row, alpha=0.05):
    pval_fdr, pval_fdr = fdr_correction(row['pvals'],
                                   alpha=alpha,
                                   method='indep')
    return pval_fdr

def get_max_decoding(row):
    scores_mean = row['scores'].mean(axis=0)
    return np.max(scores_mean)

def get_ix_max_decoding(row):
    scores_mean = row['scores'].mean(axis=0)
    return np.argmax(scores_mean)


df['scores'] = df.apply(lambda row: np.asarray(row['scores']), axis=1)
df['max_decoding'] = df.apply(lambda row: get_max_decoding(row), axis=1)
df['ix_max_decoding'] = df.apply(lambda row: get_ix_max_decoding(row), axis=1)
df['reject_fdr'] = df.apply(lambda row: get_reject_fdr(row), axis=1)
df['pval_fdr'] = df.apply(lambda row: get_pval_fdr(row), axis=1)



#print(df)
if not df.empty:
    fn_json = os.path.join(args.path2output, 'decoding_results.json')
    df.to_json(fn_json)
    print(f'JSON file saved to: {fn_json}')
else:
    print('Data frame is empty. Json file was not saved')

