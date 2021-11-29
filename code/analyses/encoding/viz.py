#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 18:55:06 2021

@author: yl254115
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_rf_coefs(results, i_channel, ch_name, feature_info, args, group=False):
    rfs = results['full']['rf_sentence'] # list of models with len=num_cv-splits
    times_rf = rfs[0].delays_*1000/rfs[0].sfreq
    # COEFs
    coefs = np.asarray([rf.coef_[i_channel, :, :] for rf in rfs])
    coefs_mean = coefs.mean(axis=0)
    coefs_std = coefs.std(axis=0)
    times_word_epoch = results['times_word_epoch']
    # Scores by time 
    scores_by_time = np.asarray([scores[:, i_channel] for scores in results['full']['scores_by_time']])
    scores_by_time_mean = scores_by_time.mean(axis=0)
    scores_by_time_std = scores_by_time.std(axis=0)
    # Total score
    total_score = np.asarray([scores[i_channel] for scores in results['full']['total_score']])
    # negative_r2 = scores_by_time_mean>0
    
    # PLOT
    fig, ax = plt.subplots(figsize=(15,10))
    ax.set_title(f'{ch_name}, $r$ = {total_score.mean():1.2f} +- {total_score.std():1.2f}', fontsize=24)
    color = 'k'
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Correlation coefficient ($r$)', color=color, fontsize=40)  # we already handled the x-label with ax1
    ax2.plot(times_word_epoch*1000, scores_by_time_mean, color=color, lw=3)    
    ax2.fill_between(times_word_epoch*1000, scores_by_time_mean+scores_by_time_std, scores_by_time_mean-scores_by_time_std, color=color, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim((0, 1)) 
    
    feature_names = feature_info.keys()
    for i_feature, feature_name in enumerate(feature_names):
        color, ls, lw, marker = get_curve_style(feature_name, feature_info)
        st, ed = feature_info[feature_name]['IXs']
        if group:
            # IX_max_abs = np.argmax(np.abs(coefs_mean[st:ed, :]), axis=0)
            coef_curr_feature = np.mean(coefs_mean[st:ed, :], axis=0)
            ax.plot(times_rf, coef_curr_feature, color=color, ls=ls, lw=lw,
                    marker=marker, markersize=15, label=feature_name)
        else:
            for i_value, feature_value in enumerate(feature_info[feature_name]['names']):
                coef_curr_feature = coefs_mean[st+i_value, :]
                ax.plot(times_rf, coef_curr_feature, color=color, ls=ls, lw=lw, label=feature_value)
    
    ax.legend(loc='center left', bbox_to_anchor=(1.12, 0, 0.3, 1), ncol=int(np.ceil(len(feature_names)/40)), fontsize=24)
    ax.set_xlabel('Time (msec)', fontsize=20)
    ax.set_ylabel(r'Beta', fontsize=20)
    ax.set_ylim((None, None)) 
    if args.block_type == 'visual':
        ax.axvline(x=0, ls='--', color='k')
        ax.axvline(x=500, ls='--', color='k')
    ax.axhline(ls='--', color='k')    
    ax.tick_params(axis='both', labelsize=18)
    ax2.tick_params(axis='both', labelsize=18)
    plt.subplots_adjust(right=0.65)
    
    return fig


def get_scores_by_time(results, i_channel, feature_name):
    scores_by_time = np.asarray([scores[:, i_channel] for scores in results[feature_name]['scores_by_time']])
    scores_by_time_mean = scores_by_time.mean(axis=0)
    scores_by_time_std = scores_by_time.std(axis=0)
    n_samples = scores_by_time.shape[0]
    scores_by_time_sem = scores_by_time_std/np.sqrt(n_samples)
    return scores_by_time_mean, scores_by_time_sem


def plot_rf_r2(results, i_channel, ch_name, feature_info, args):
    fig, ax = plt.subplots(figsize=(15,10))

    # time points and total score
    times_word_epoch = results['times_word_epoch']
    total_score = np.asarray([scores[i_channel] for scores in results['full']['total_score']])

    # Scores by time (full model)
    scores_by_time_full_mean, scores_by_time_full_sem = \
            get_scores_by_time(results, i_channel, 'full')
    
    # Draw full-model results
    #ax.set_title(f'{ch_name}, $r$ = {total_score.mean():1.2f} +- {total_score.std():1.2f}', fontsize=24)
    color = 'k'
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Correlation coefficient ($r$)', color=color, fontsize=40)  
    ax2.plot(times_word_epoch*1000, scores_by_time_full_mean, color=color, lw=3)
    ax2.fill_between(times_word_epoch*1e3,
                     scores_by_time_full_mean+scores_by_time_full_sem,
                     scores_by_time_full_mean-scores_by_time_full_sem,
                     color=color,
                     alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim((0, 1)) 
    ax2.set_xlim((-100, 600)) 

    
    feature_names = []  # performance of the full model must be calculated
    if args.each_feature_value:
        for f in feature_info.keys():
            for f_name in feature_info[f]['names']:
                feature_names.append(f_name)
    else:
        feature_names = feature_info.keys()
    
    for i_feature, feature_name in enumerate(feature_names):
        scores_by_time_feature_mean, scores_by_time_feature_sem = \
            get_scores_by_time(results, i_channel, feature_name)
        color, ls, lw, marker = get_curve_style(feature_name, feature_info)
        ax.plot(times_word_epoch*1000,
                scores_by_time_full_mean - scores_by_time_feature_mean,
                color=color, ls=ls, lw=lw,
                marker=marker, markersize=15, label=feature_name)
        #ax2.fill_between(times_word_epoch*1e3,
        #                 scores_by_time_feature_mean+scores_by_time_feature_sem,
        #                 scores_by_time_feature_mean-scores_by_time_feature_sem,
        #                 color=color,
        #                 alpha=0.2)
        #print(feature_name, color)
        
        #diff = scores_by_time_full - scores_by_time_curr_feature 
        #diff_mean = diff.mean(axis=0) 
        #diff_std = diff.mean(axis=0)
        #ax.plot(times_word_epoch*1000, diff_mean, color=color, ls=ls, lw=lw,
        #        marker=marker, markersize=15, label=feature_name)
        #ax.fill_between(times_word_epoch*1e3, diff_mean + diff_std, diff_mean - diff_std , color=color, alpha=0.2)
    
    #ax.legend(loc='center left', bbox_to_anchor=(1.12, 0, 0.3, 1), ncol=int(np.ceil(len(feature_names)/40)), fontsize=24)
    ax.set_xlabel('Time (msec)', fontsize=40)
    ax.set_ylabel(r'$\Delta r$', fontsize=40)
    ax.set_ylim((0, 0.3))
    if args.block_type == 'visual':
        ax.axvline(x=0, ls='--', color='k')
        ax.axvline(x=500, ls='--', color='k')
    ax.axhline(ls='--', color='k')    
    ax.tick_params(axis='both', labelsize=35)
    ax2.tick_params(axis='both', labelsize=35)
    #plt.subplots_adjust(right=0.65)
    
    return fig


def get_curve_style(feature_name, feature_info):
    marker = None
    # CHECK IF IT'S A FEATURE NAME OR FEATURE-VALUE NAME
    if feature_name in feature_info.keys():  # is feature name
        color = feature_info[feature_name]['color']
        if feature_name == 'semantics':
            color = 'xkcd:orange'
            color = 'orange'
        f_name = feature_name
    else:  # is value name
        f_name = None
        for k in feature_info.keys():
            if feature_name in feature_info[k]['names']:
                f_name = k
                break
        if not f_name:
            raise(f'Unrecognized feature name {feature_name}')
        n_values = len(feature_info[f_name]['names'])
        colors = plt.cm.jet(np.linspace(0,1,n_values))
        IX = feature_info[f_name]['names'].index(feature_name)
        color = colors[IX]
        if 'letter-' in feature_name:
            marker = f'${feature_name[-1]}$'
    
    if ('ls' in feature_info[f_name].keys()) and feature_info[f_name]['ls']:
        ls = feature_info[f_name]['ls']
    else:
        ls = '-'
    if ('lw' in feature_info[f_name].keys()) and feature_info[f_name]['lw']:
        lw = feature_info[f_name]['lw']
    else:
        lw = 3
    
    return color, ls, lw, marker
