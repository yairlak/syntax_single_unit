#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 18:55:06 2021

@author: yl254115
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_rf_coefs(results, i_channel, ch_name, feature_info, args):
    rf = results['full']['rf_sentence']['split-0']
    times_rf = rf.delays_*1000/rf.sfreq
    coefs = rf.coef_[i_channel, :, :]
    times_word_epoch = results['times_word_epoch']
    scores_by_time = results['full']['scores_by_time']['split-0'][:, i_channel]
    total_score = results['full']['total_score']['split-0'][i_channel]
    
    
    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_title(f'{ch_name}, $R^2$ = {total_score:1.2f}', fontsize=24)
    color = 'k'
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Coefficient of determination ($R^2$)', color=color, fontsize=20)  # we already handled the x-label with ax1
    ax2.plot(times_word_epoch*1000, scores_by_time, color=color, lw=3)
    # ax2.fill_between(times*1e3, r2_full_mean+r2_full_std, r2_full_mean-r2_full_std, color=color, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim((0, 1)) 
    
    feature_names = feature_info.keys()
    for i_feature, feature_name in enumerate(feature_names):
        IXs = feature_info[feature_name]['IXs']
        coef_curr_feature = np.max(coefs[IXs[0]:IXs[1], :], axis=0)
        
        color, ls, lw = get_curve_style(feature_name, feature_info)
        
        ax.plot(times_rf, coef_curr_feature, color=color, ls=ls, lw=lw, label=feature_name)
        # ax.fill_between(times*1e3, effect_size_mean + effect_size_std, effect_size_mean - effect_size_std , color=color, alpha=0.2)
    
    ax.legend(loc='center left', bbox_to_anchor=(1.12, 0.5), ncol=int(np.ceil(len(feature_names)/40)))
    ax.set_xlabel('Time (msec)', fontsize=20)
    ax.set_ylabel(r'Beta', fontsize=20)
    ax.set_ylim((None, None)) 
    if args.block_type == 'visual':
        ax.axvline(x=0, ls='--', color='k')
        ax.axvline(x=500, ls='--', color='k')
    ax.axhline(ls='--', color='k')    
    plt.subplots_adjust(right=0.8)
    
    return fig



def plot_rf_r2(results, i_channel, ch_name, feature_info, args):
    times_word_epoch = results['times_word_epoch']
    total_score = results['full']['total_score']['split-0'][i_channel]
    scores_by_time_full = results['full']['scores_by_time']['split-0'][:, i_channel]
    
    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_title(f'{ch_name}, $R^2$ = {total_score:1.2f}', fontsize=24)
    color = 'k'
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Coefficient of determination ($R^2$)', color=color, fontsize=20)  # we already handled the x-label with ax1
    ax2.plot(times_word_epoch*1000, scores_by_time_full, color=color, lw=3)
    # ax2.fill_between(times*1e3, r2_full_mean+r2_full_std, r2_full_mean-r2_full_std, color=color, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim((0, 1)) 
    
    feature_names = feature_info.keys()
    for i_feature, feature_name in enumerate(feature_names):
        scores_by_time_curr_feature = results[feature_name]['scores_by_time']['split-0'][:, i_channel]
        
        color, ls, lw = get_curve_style(feature_name, feature_info)
        
        ax.plot(times_word_epoch*1000, scores_by_time_full-scores_by_time_curr_feature, color=color, ls=ls, lw=lw, label=feature_name)
        # ax.fill_between(times*1e3, effect_size_mean + effect_size_std, effect_size_mean - effect_size_std , color=color, alpha=0.2)
    
    ax.legend(loc='center left', bbox_to_anchor=(1.12, 0.5), ncol=int(np.ceil(len(feature_names)/40)))
    ax.set_xlabel('Time (msec)', fontsize=20)
    ax.set_ylabel(r'Beta', fontsize=20)
    ax.set_ylim((0, None)) 
    if args.block_type == 'visual':
        ax.axvline(x=0, ls='--', color='k')
        ax.axvline(x=500, ls='--', color='k')
    ax.axhline(ls='--', color='k')    
    plt.subplots_adjust(right=0.8)
    
    return fig


def get_curve_style(feature_name, feature_info):
    if feature_info[feature_name]['color']:
        color = feature_info[feature_name]['color']
    else:
        color = None
    if ('ls' in feature_info[feature_name].keys()) and feature_info[feature_name]['ls']:
        ls = feature_info[feature_name]['ls']
    else:
        ls = '-'
    if ('lw' in feature_info[feature_name].keys()) and feature_info[feature_name]['lw']:
        lw = feature_info[feature_name]['lw']
    else:
        lw = 3
    
    return color, ls, lw