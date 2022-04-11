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
parser.add_argument('--patient', action='append', default=['502'],
                    help='Patient string')
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike'],
                    action='append', default=['spike'], help='electrode type')
parser.add_argument('--filter', action='append',
                    default=['raw'],
                    help='raw/high-gamma')
parser.add_argument('--smooth', default=50, type=int,
                    help='Gaussian smoothing in msec')
parser.add_argument('--decimate', default=50, type=int,
                    help='If not empty, decimate data for speed.')
parser.add_argument('--probe-name', default=None, nargs='*',
                    action='append', type=str,
                    help='Probe name to plot (ignores channel-name/num)')
parser.add_argument('--channel-name', default=[], nargs='*', action='append',
                    type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=None, nargs='*', action='append',
                    type=int, help='channel number (if empty all channels)')
# FEATURES
parser.add_argument('--feature-list',
                    nargs='*',
#                    action='append',
                    default=['orthography_pos', 'position'],
                    #default=None,
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
    if ch_name not in ['GB4-RFSG1_57p1', 'GB4-RFSG6_62p1']:
        continue
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
    

    # PLOT
    
    # ax.set_title(f'{ch_name}', fontsize=24)
    
    
    # color = 'k'
    # ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_ylabel('Correlation coefficient ($r$)', color=color, fontsize=30)
    # ax2.plot(times, scores_by_time['full'], color=color, lw=3)
    # ax2.fill_between(times,
    #                 scores_by_time['full'] + sem_by_time['full'],
    #                 scores_by_time['full'] - sem_by_time['full'],
    #                 color=color,
    #                 alpha=0.2)
    # ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_xlim((-0.250, 0.750))
    # ax2.set_ylim((0,1))
    
    alphabet=[letter for letter in 'abcdefghijklmnopqrstuvwxyz']
    positions = ['First', 'Middle', 'Last']
    
    n_letters = len(alphabet)
    feature_names = feature_info.keys()
    
    for i_t, t in enumerate(times):
        feature_name = 'orthography_pos'
        st, ed = feature_info[feature_name]['IXs']
            
        coef_curr_feature = coefs_mean['full'][:, st:ed]
        
        if t<0.1 or t>0.5:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(10,10))
        
        ax.set_xlim((0, 2.5))
        ax.set_ylim((0, 26))
        # axs[0].set_xticks(range(len(positions)))
        # axs[0].set_xticklabels(positions)
        # axs[0].set_yticks(range(len(alphabet)))
        # axs[0].set_yticklabels(alphabet)
        plt.axis('off')
        
        
        for i_pos, pos in enumerate(positions):
            ax.text(i_pos, -1, pos.capitalize() + ' letter', fontsize=26)
            for i_letter, letter in enumerate(alphabet):
                coef = coef_curr_feature[i_t, i_pos*n_letters + i_letter]
                color = 'r' if coef>0 else 'b'
                fontsize = np.abs(coef*1e3*5)
                print(i_pos, pos, i_letter, letter, color, fontsize)
                ax.text(i_pos, i_letter, letter,
                            fontsize=fontsize,
                            color=color)
        fname_fig = os.path.join(args.path2figures, 
                              args.patient[0],
                              args.data_type[0],
                              'evoked_coef_letter_by_position_' +
                              fname + f'_{ch_name}_t_{t}.png')
        fig.savefig(fname_fig)
        plt.close(fig)
        print('Figure saved to: ', fname_fig)
                
        ############
        # BAR PLOT #
        ############
        fig, ax = plt.subplots(1, 1, figsize=(10,10))
        coef_first_pos_mean = np.abs(coef_curr_feature[i_t, 0:n_letters]).mean()
        coef_first_pos_sem = coef_curr_feature[i_t, 0:n_letters].std()/np.sqrt(n_letters)
        coef_second_pos_mean = np.abs(coef_curr_feature[i_t, n_letters:2*n_letters]).mean()
        coef_second_pos_sem = coef_curr_feature[i_t, n_letters:2*n_letters].std()/np.sqrt(n_letters)
        coef_third_pos_mean = np.abs(coef_curr_feature[i_t, 2*n_letters:3*n_letters]).mean()
        coef_third_pos_sem = coef_curr_feature[i_t, 2*n_letters:3*n_letters].std()/np.sqrt(n_letters)
        coef_word_length = coef_curr_feature[i_t, -1]
        coef_letters = np.abs(coef_curr_feature[:-1]).mean()
        
        feature_name = 'position'
        st, ed = feature_info[feature_name]['IXs']
        coef_curr_feature = coefs_mean['full'][:, st:ed]
        coef_is_first_word = coef_curr_feature[i_t, 0]
        coef_word_position = coef_curr_feature[i_t, 1]
        coef_is_last_word = coef_curr_feature[i_t, 2]
        
        #labels = #[pos.capitalize() + ' letter' for pos in positions] + \
            # [coef_first_pos_mean,
            #                 coef_second_pos_mean,
            #                 coef_third_pos_mean,
            #                 coef_word_length,
            #                 coef_is_first_word,
            #                 coef_word_position,
            #                 coef_is_last_word]
        heights = [coef_letters, coef_word_length, coef_word_position]
        labels = ['Letters', 'Word length', 'Word position']
        ax.bar(labels, heights)
        
        fname_fig = os.path.join(args.path2figures, 
                              args.patient[0],
                              args.data_type[0],
                              'evoked_coef_letter_by_position_barplot_' +
                              fname + f'_{ch_name}_t_{t}.png')
        fig.savefig(fname_fig)
        plt.close(fig)
        print('Figure saved to: ', fname_fig)
    
     #ax.plot(times, coef_curr_feature, color=color, ls=ls, lw=lw, label=feature_info[feature_name]['names'])
    # ax.plot(times, coef_curr_feature, ls=ls, lw=lw, label=feature_info[feature_name]['names'])
    # print(feature_name)

    #ax.legend(loc='center left', bbox_to_anchor=(1.5, 0, 0.5, 1.2), ncol=int(np.ceil(len(feature_names)/10)), fontsize=16)
    # ax.legend(loc='center left', bbox_to_anchor=(1.2, 0, 0.5, 1), ncol=int(np.ceil(coefs_mean['full'].shape[1]/30)), fontsize=16)
    # ax.set_xlabel('Time (msec)', fontsize=30)
    # ax.set_ylabel(r'Beta', fontsize=30)
    # ax.set_ylim((None, None)) 
    # if args.block_type == 'visual':
    #     ax.axvline(x=0, ls='--', color='k')
    #     ax.axvline(x=500, ls='--', color='k')
    # ax.axhline(ls='--', color='k')    
    # # ax.set_xlim((-0.250, 0.750))
    # ax.tick_params(axis='both', labelsize=18)
    # #ax2.tick_params(axis='both', labelsize=18)
    # plt.subplots_adjust(right=0.45)

        
    
    
