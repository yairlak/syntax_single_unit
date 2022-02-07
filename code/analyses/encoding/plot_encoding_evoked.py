#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:48:59 2021

@author: yl254115
"""
import argparse
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import sys
import pickle
from viz import plot_evoked_coefs, plot_evoked_r
sys.path.append('..')
from utils.utils import dict2filename
import matplotlib.pyplot as plt
import numpy as np
from mne.stats import fdr_correction

parser = argparse.ArgumentParser(description='Plot TRF results')
# DATA
parser.add_argument('--patient', action='append', default=[],
                    help='Patient string')
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike'],
                    action='append', default=[], help='electrode type')
parser.add_argument('--filter', action='append',
                    default=[],
                    help='raw/high-gamma')
parser.add_argument('--smooth', default=50, type=int,
                    help='Gaussian smoothing in msec')
parser.add_argument('--decimate', default=50, type=int,
                    help='If not empty, decimate data for speed.')
parser.add_argument('--probe-name', default=None, nargs='*',
                    action='append', type=str,
                    help='Probe name to plot (ignores channel-name/num)')
parser.add_argument('--channel-name', default=None, nargs='*', action='append',
                    type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=None, nargs='*', action='append',
                    type=int, help='channel number (if empty all channels)')
# FEATURES
parser.add_argument('--feature-list',
                    nargs='*',
#                    action='append',
                    default=None,
                    # default=['is_first_word', 'positional', 'orthography', 'lexicon', 'syntax', 'semantics', 'is_last_word'],
                    # default = ['position', 'orthography', 'lexicon', 'syntax', 'semantics'],
                    #default = ['orthography'],
                    #default = 'is_first_word word_onset positional orthography lexicon syntax semantics'.split(),
                    help='Feature to include in the encoding model')
parser.add_argument('--each-feature-value', default=False, action='store_true',
                    help="Evaluate model after ablating each feature value. \
                         If false, ablate all feature values together")
parser.add_argument('--keep', default=False, action='store_true',
                    help="If True, plot the case for which the feature was. \
                          kept as the only one instead of removed from the model")
# MODEL
parser.add_argument('--model-type', default='ridge',
                    choices=['ridge', 'lasso', 'ridge_laplacian', 'standard'])
parser.add_argument('--ablation-method', default='remove',
                    choices=['shuffle', 'remove', 'zero'],
                    help='Method used to calcuated feature importance\
                        by reducing/ablating a feature family')

# MISC
parser.add_argument('--path2output',
                    default=os.path.join('..', '..', '..',
                                         'Output', 'encoding_models'))
parser.add_argument('--path2figures',
                    default=os.path.join('..', '..', '..',
                                         'Figures', 'encoding_models'))
parser.add_argument('--query-train', default="block in [1,3,5] and word_length>1")
parser.add_argument('--query-test', default="block in [1,3,5] and word_length>1")


#############
# USER ARGS #
#############
args = parser.parse_args()
assert len(args.patient) == len(args.data_type) == len(args.filter)
args.patient = ['patient_' + p for p in args.patient]
args.block_type = 'both'
if not args.query_test:
    args.query_test = args.query_train

# args.feature_list = ['is_first_word', 'word_onset']+'positional_orthography_lexicon_syntax'.split('_') 
# args.feature_list = ['is_first_word', 'word_onset']+'positional_orthography'.split('_') 



print('args\n', args)
list_args2fname = ['patient', 'data_type', 'filter', 'decimate', 'smooth',
                   'model_type', 'probe_name', 'ablation_method',
                   'query_train', 'feature_list', 'each_feature_value']
                   #'query_train', 'feature_list', 'each_feature_value']
if args.query_train != args.query_test:
    list_args2fname.extend(['query_test'])

if not os.path.exists(os.path.join(args.path2figures, args.patient[0], args.data_type[0])):
    os.makedirs(os.path.join(args.path2figures, args.patient[0], args.data_type[0]))

args2fname = args.__dict__.copy()
fname = dict2filename(args2fname, '_', list_args2fname, '', True)

#########################
# LOAD ENCODING RESULTS #
#########################
results, ch_names, args_evoked, feature_info = \
    pickle.load(open(os.path.join(args.path2output, 'evoked_' + fname + '.pkl'), 'rb'))
print(args_evoked)

n_channels, n_times = results['full']['scores_by_time_False'].shape

keep = False
# nd-array of size n_times
times = results['times']
# dict with feature-name keys, 
# each contains a list (len=n_splits) with n_channels X n_times results array
scores_by_time_per_split = {k:results[k][f'scores_by_time_per_split_{keep}']
                            for k in results.keys() if not k.startswith('times')}
coefs_by_time_per_split = {k:results[k][f'model_per_split_{keep}']
                           for k in results.keys() if not k.startswith('times')}


stats_by_time = {k:results[k][f'stats_by_time_{keep}']
                     for k in results.keys() if not k.startswith('times')}
    

# dict with feature-name keys, 
# each contains a n_channels X n_times results array


#######
# FDR #
#######
alpha = 0.05
reject_fdr = {}
for k in stats_by_time.keys():    
    reject_fdr[k], stats_by_time[k] = fdr_correction(stats_by_time[k],
                                                  alpha=alpha,
                                                  method='indep')

############
# PLOTTING #
############
#n_splits = len(scores_by_time_per_split['full'])

for i_channel, ch_name in enumerate(ch_names):
    #scores_mean = {k:np.asarray(scores_by_time_per_split[k]).mean(axis=0)[i_channel, :]
    #               for k in scores_by_time_per_split.keys()}
    #scores_sem = {k:np.asarray(scores_by_time_per_split[k]).std(axis=0)[i_channel, :]/np.sqrt(n_splits)
    #              for k in scores_by_time_per_split.keys()}
    scores_by_time = {k:results[k][f'scores_by_time_{keep}'][i_channel, :]
                      for k in results.keys() if not k.startswith('times')}
    stats_by_time = {k:results[k][f'stats_by_time_{keep}'][i_channel, :]
                     for k in results.keys() if not k.startswith('times')}
    sem_by_time = {k:np.zeros_like(results[k][f'stats_by_time_{keep}'][i_channel, :])
                     for k in results.keys() if not k.startswith('times')}
    
    coefs_mean , coefs_sem= {}, {}
    for k in coefs_by_time_per_split.keys():
        coefs_all_splits = []
        for i_model, model in enumerate(coefs_by_time_per_split[k]):
            coefs_all_splits.append(model.coef_.reshape(n_channels, n_times, -1)[i_channel, :, :])
        coefs_all_splits = np.dstack(coefs_all_splits)
        coefs_mean[k] = coefs_all_splits.mean(axis=2)
        coefs_sem[k] = coefs_all_splits.std(axis=2)/np.sqrt(i_model+1)
                   
    
    
    reject_fdr_curr_channel = {k:reject_fdr[k][i_channel, :]
                               for k in reject_fdr.keys()}
    
    fig_r2 = plot_evoked_r(times,
                           scores_by_time,
                           sem_by_time,
                           reject_fdr_curr_channel, # after FDR correction
                           ch_name, feature_info, args, keep)

    fname_fig = os.path.join(args.path2figures, 
                             args.patient[0],
                             args.data_type[0],
                             'evoked_r_' +
                             fname + f'_{ch_name}_keep_{keep}.png')
    fig_r2.savefig(fname_fig)
    plt.close(fig_r2)
    print('Figure saved to: ', fname_fig)


    for group in [False, True]:
        fig_coef = plot_evoked_coefs(times,
                                     coefs_mean,
                                     coefs_sem,
                                     scores_by_time,
                                     sem_by_time,
                                     reject_fdr_curr_channel, # after FDR correction
                                     ch_name, feature_info, args, keep, group)
    
        fname_fig = os.path.join(args.path2figures, 
                                 args.patient[0],
                                 args.data_type[0],
                                 'evoked_coef_' +
                                 fname + f'_{ch_name}_keep_{keep}_group_{group}.png')
        fig_coef.savefig(fname_fig)
        plt.close(fig_coef)
        print('Figure saved to: ', fname_fig)
    
