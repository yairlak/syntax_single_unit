#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 18:55:06 2021

@author: yl254115
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import sem, ttest_ind

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_rf_coefs(results, i_channel, ch_name, feature_info, args, group=False):
    rfs = results['full']['rf_sentence_per_split'] # list of models with len=num_cv-splits
    times_rf = rfs[0].delays_*1000/rfs[0].sfreq
    # COEFs
    coefs = np.asarray([rf.coef_[i_channel, :, :] for rf in rfs])
    coefs_mean = coefs.mean(axis=0)
    coefs_std = coefs.std(axis=0)
    times_word_epoch = results['times_word_epoch']
    # Scores by time 
    scores_by_time = np.asarray([scores[i_channel, :] for scores in results['full']['rs_word_per_split']])
    scores_by_time_mean = scores_by_time.mean(axis=0)
    scores_by_time_std = scores_by_time.std(axis=0)
    # Total score
    total_score = np.asarray([scores[i_channel] for scores in results['full']['rs_sentence_per_split']])
    # negative_r2 = scores_by_time_mean>0
    
    # PLOT
    fig, ax = plt.subplots(figsize=(35,20))
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
            cmap = get_cmap(len(feature_info[feature_name]['names']))
            for i_value, feature_value in enumerate(feature_info[feature_name]['names']):
                coef_curr_feature = coefs_mean[st+i_value, :]
                ax.plot(times_rf, coef_curr_feature, color=cmap(i_value), ls=ls, lw=lw, label=feature_value)
    
    #ax.legend(loc='center left', bbox_to_anchor=(1.12, 0, 0.3, 1), ncol=int(np.ceil(len(feature_names)/20)), fontsize=24)
    ax.legend(loc='center left', bbox_to_anchor=(1.12, 0, 0.3, 1), ncol=3, fontsize=24)
    ax.set_xlabel('Time (msec)', fontsize=20)
    ax.set_ylabel(r'Beta', fontsize=20)
    ax.set_ylim((None, None)) 
    if args.block_type == 'visual':
        ax.axvline(x=0, ls='--', color='k')
        ax.axvline(x=500, ls='--', color='k')
    ax.axhline(ls='--', color='k')    
    ax.tick_params(axis='both', labelsize=18)
    ax2.tick_params(axis='both', labelsize=18)
    plt.subplots_adjust(right=0.5)
    
    return fig


def get_scores_by_time(results, i_channel, feature_name):
    scores_by_time = np.asarray([scores[i_channel, :] for scores in results[feature_name]['rs_word_per_split']])
    scores_by_time_mean = scores_by_time.mean(axis=0)
    scores_by_time_std = scores_by_time.std(axis=0)
    n_samples = scores_by_time.shape[0]
    scores_by_time_sem = scores_by_time_std/np.sqrt(n_samples)
    return scores_by_time_mean, scores_by_time_sem


def plot_rf_r2(results, i_channel, ch_name, feature_info, args):
    fig, ax = plt.subplots(figsize=(15,10))

    # time points and total score
    times_word_epoch = results['times_word_epoch']
    # total_score = np.asarray([scores[i_channel] for scores in results['full']['rs_sentence']])

    # Scores by time (full model)
    scores_by_time_full_mean, scores_by_time_full_sem = \
            get_scores_by_time(results, i_channel, 'full')
    
    # Draw full-model results
    #ax.set_title(f'{ch_name}, $r$ = {total_score.mean():1.2f} +- {total_score.std():1.2f}', fontsize=24)
    color = 'k'
    #ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    #ax2.set_ylabel('Correlation coefficient ($r$)', color=color, fontsize=40)  
    #ax2.plot(times_word_epoch*1000, scores_by_time_full_mean, color=color, lw=3)
    #ax2.fill_between(times_word_epoch*1e3,
    #                 scores_by_time_full_mean+scores_by_time_full_sem,
    #                 scores_by_time_full_mean-scores_by_time_full_sem,
    #                 color=color,
    #                 alpha=0.2)
    #ax2.tick_params(axis='y', labelcolor=color)
    #ax2.set_ylim((0, 1)) 
    #ax2.set_xlim((-100, 600)) 

    
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
                marker=marker, markersize=30, label=feature_name)
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
    
    # ax.legend(loc='center left', bbox_to_anchor=(1.12, 0, 0.3, 1), ncol=int(np.ceil(len(feature_names)/40)), fontsize=24)
    ax.set_xlabel('Time (msec)', fontsize=40)
    ax.set_ylabel(r'$\Delta r$', fontsize=40)
    ax.set_ylim((0, 0.1))
    if args.block_type == 'visual':
        ax.axvline(x=0, ls='--', color='k')
        ax.axvline(x=500, ls='--', color='k')
    ax.axhline(ls='--', color='k')    
    ax.tick_params(axis='both', labelsize=35)
    #ax2.tick_params(axis='both', labelsize=35)
    # plt.subplots_adjust(right=0.5)
    
    return fig



def plot_rf_bar_r2(results, i_channel, ch_name, feature_info, args):
    fig, ax = plt.subplots(figsize=(15,10))

    
    # Scores by time (full model)
    rs_full_per_split = [rs[i_channel] for rs in results['full']['rs_sentence_per_split']]

    feature_names = sorted(list(set(results) - set(['full', 'times_word_epoch'])))
    feature_names = [feature_names[i] for i in [0, 3, 2, 1, 4]]
    feature_importances_mean, feature_importances_sem, colors, ps = [], [], [], []
    for i_feature, feature in enumerate(feature_names):
        rs_feature_per_split = [rs[i_channel] for rs in results[feature]['rs_sentence_per_split']]
        dr_per_split = [r_full - r_feature for r_full, r_feature in zip(rs_full_per_split, rs_feature_per_split)]
        feature_importances_mean.append(np.mean(dr_per_split))
        feature_importances_sem.append(sem(dr_per_split))
        
        # VIZ
        color, ls, lw, marker = get_curve_style(feature, feature_info)
        colors.append(color)
        t, p = ttest_ind(rs_feature_per_split,
                         rs_full_per_split)
        ps.append(p)
        
    y_pos = np.arange(len(feature_names))
    barplot = ax.barh(y_pos, feature_importances_mean, xerr=feature_importances_sem, color=colors)#, align='center')
    ax.set_yticks(y_pos)#, labels=feature_names)
    ax.set_yticklabels(feature_names)#, labels=feature_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    
    
    for patch, p in zip(barplot.patches,ps):
        if p<0.05:
            ax.text(1.1*patch.get_width(),
                    patch.get_y()+patch.get_height(), 
                         '*', ha='center', fontsize=20)
            # barplot.axvline(p.get_x() + p.get_width() / 2., lw=0.5)
    # for y in ypos:
    #     if ps[y] < 0.05:
    #         ax.text(y, )
    
    # ax.set_xlim((0, 1.5*np.max(feature_importances_mean)))
    # ax.set_xticks([0, np.max(feature_importances_mean)])
    ax.set_xlabel('Feature importance', fontsize=40)
    
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.set_xlim((0, None))
    ax.tick_params(axis='both', labelsize=24)
    plt.subplots_adjust(left=0.4)

    
    return fig


def get_curve_style(feature_name, feature_info):
    marker = None
    # CHECK IF IT'S A FEATURE NAME OR FEATURE-VALUE NAME
    if feature_name in feature_info.keys():  # is feature name
        color = feature_info[feature_name]['color']
        # if feature_name == 'semantics':
        #     color = 'xkcd:orange'
        #     color = 'orange'
        
        
       
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
        if 'Phone-' in feature_name:
            m = feature_name.split('-')[-1]
            marker = f'${m}$'
    
    if ('ls' in feature_info[f_name].keys()) and feature_info[f_name]['ls']:
        ls = feature_info[f_name]['ls']
    else:
        ls = '-'
    if ('lw' in feature_info[f_name].keys()) and feature_info[f_name]['lw']:
        lw = feature_info[f_name]['lw']
    else:
        lw = 3
        
    if feature_name == 'position':
        #dict_prop['color'] = 'grey'
        ls = 'dashed'
        #dict_prop['lw'] = 3

    # PHONOLOGY
    if feature_name in ['phonology', 'phonemes']:
        #dict_prop['color'] = 'm'
        ls = 'dashdot'
        
        #dict_prop['lw'] = 3

    # ORTHOGRAPHY
    if feature_name == 'orthography':
        #dict_prop['color'] = 'r'
        ls = 'dashdot'
        #dict_prop['lw'] = 3

    # LEXICON
    if feature_name == 'lexicon':
        ls = 'dotted'
        
    # SEMANTICS
    if feature_name == 'semantics':
        ls = 'solid'
        
    # SYNTAX
    if feature_name == 'syntax':
        ls = (0, (3, 5, 1, 5, 1, 5)) #dashdotdotted
        
    
    return color, ls, lw, marker


def plot_evoked_r(times, scores_mean, scores_sem, reject_fdr,
              ch_name, feature_info, args, y_lim=None, keep=False):
    fig, ax = plt.subplots(figsize=(20,10))
    
    # Draw full-model results
    # scores_full_mean = scores['full']['scores_by_time'][0][i_channel, :]
    # print(scores_full_mean)
    # print(scores_full_mean.shape)
    color = 'k'
    # ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_ylabel('Correlation coefficient ($r$)', color=color, fontsize=40)
    # ax2.plot(times*1e3, scores_mean['full'], color=color, lw=3)
    # ax2.fill_between(times*1e3,
    #                 scores_mean['full'] + scores_sem['full'],
    #                 scores_mean['full'] - scores_sem['full'],
    #                 color=color,
    #                 alpha=0.2)
    # ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_xlim((-250, 750))
    
    if any(reject_fdr['full']):
        sig_period = False
        for i_t, reject in enumerate(reject_fdr['full']):
            if reject and (not sig_period): # Entering a significance zone
                t1 = times[i_t]
                sig_period = True
            elif (not reject) and sig_period: # Exiting a sig zone
                t2 = times[i_t-1]
                #ax.axvspan(t1, t2, facecolor='g', alpha=0.2)
                # ax2.hlines(y=1, xmin=t1*1e3, xmax=t2*1e3,
                #            linewidth=8, color='k', alpha=0.3)
                sig_period = False
            elif sig_period and (i_t==len(reject_fdr)-1): # Last time point
                t2 = times[i_t]
                # ax2.hlines(y=1, xmin=t1*1e3, xmax=t2*1e3,
                #            linewidth=8, color='k', alpha=0.3)

    feature_names = []  # performance of the full model must be calculated
    if args.each_feature_value:
        for f in feature_info.keys():
            for f_name in feature_info[f]['names']:
                feature_names.append(f_name)
    else:
        feature_names = feature_info.keys()
    
    n_features = len(feature_names)

    if y_lim is None: 
        y_lim = 1 + (1+n_features)*0.02
        # ax2.set_ylim((0, y_lim))
        y_lim = 0.05 +(1+n_features)*0.02
    # else:
    #     ax2.set_ylim((0, 0.6))
    
    for i_feature, feature_name in enumerate(feature_names):
        color, ls, lw, marker = get_curve_style(feature_name, feature_info)
        mask_sig = np.zeros_like(reject_fdr[feature_name])
        if any(reject_fdr[feature_name]):
            sig_period = False
            for i_t, reject in enumerate(reject_fdr[feature_name]):
                if reject and (not sig_period): # Entering a significance zone
                    t1 = times[i_t]
                    sig_period = True
                    i_t_st = i_t
                elif (not reject) and sig_period: # Exiting a sig zone
                    t2 = times[i_t-1]
                    i_t_ed = i_t
                    #ax.axvspan(t1, t2, facecolor='g', alpha=0.2)
                    ax.hlines(y=1.02+i_feature*0.02, xmin=t1*1e3, xmax=t2*1e3,
                               linewidth=8, color=color, alpha=0.3)
                    sig_period = False
                    mask_sig[i_t_st:i_t_ed] = 1
                elif sig_period and (i_t==len(reject_fdr)-1): # Last time point
                    t2 = times[i_t]
                    i_t_ed = -1
                    ax.hlines(y=y_lim+i_feature*0.02, xmin=t1*1e3, xmax=t2*1e3,
                               linewidth=8, color=color, alpha=0.3)
                    mask_sig[i_t_st:i_t_ed] = 1

        
        
        if keep:
            # feature_importance = scores_mean[feature_name]*(2*scores_mean[feature_name])/(scores_mean['full'] + scores_mean[feature_name])
            feature_importance = scores_mean[feature_name]
        else:
            #feature_importance = scores_mean[feature_name]*(scores_mean['full'] - scores_mean[feature_name])/(scores_mean['full'] + scores_mean[feature_name])
            feature_importance = (scores_mean['full'] - scores_mean[feature_name])
        feature_importance = np.maximum(feature_importance, np.zeros_like(feature_importance))
        # feature_importance = np.minimum(feature_importance, 2*np.ones_like(feature_importance))
        # feature_importance[~mask_sig] = 0
        #print(feature_name, color, ls, lw, marker, feature_importance)
        ax.plot(times*1e3,
                feature_importance,
                color=color, ls=ls, lw=lw,
                marker=marker, markersize=25, label=feature_name)
        
    
    ax.set_xlabel('Time (msec)', fontsize=40)
    # ax.set_ylabel(r'$\Delta r$', fontsize=40)
    ax.set_ylabel('Feature importance', fontsize=40)
    #ax.set_ylim((0, y_lim+(1+n_features)*0.02))
    ax.set_ylim((0, y_lim))
    if args.block_type == 'visual':
        ax.axvline(x=0, ls='--', color='k')
        ax.axvline(x=500, ls='--', color='k')
    ax.axhline(ls='--', color='k')
    ax.set_xlim((0, 600))
    ax.tick_params(axis='both', labelsize=35)
    # ax2.tick_params(axis='both', labelsize=35)
    ax.legend(loc='center left', bbox_to_anchor=(1.5, 0, 0.5, 1.2), ncol=int(np.ceil(len(feature_names)/20)), fontsize=16)
    plt.subplots_adjust(right=0.5)

    return fig


def plot_evoked_coefs(times, coefs_mean, coefs_sem, scores_mean, scores_sem, reject_fdr_curr_channel, ch_name, feature_info, args, keep, group=False):
    
    # PLOT
    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_title(f'{ch_name}', fontsize=24)
    
    
    color = 'k'
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Correlation coefficient ($r$)', color=color, fontsize=30)
    ax2.plot(times, scores_mean['full'], color=color, lw=3)
    ax2.fill_between(times,
                    scores_mean['full'] + scores_sem['full'],
                    scores_mean['full'] - scores_sem['full'],
                    color=color,
                    alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xlim((-0.1, 0.600))
    ax2.set_ylim((0,1))
    
    # if any(reject_fdr['full']):
    #     sig_period = False
    #     for i_t, reject in enumerate(reject_fdr['full']):
    #         if reject and (not sig_period): # Entering a significance zone
    #             t1 = times[i_t]
    #             sig_period = True
    #         elif (not reject) and sig_period: # Exiting a sig zone
    #             t2 = times[i_t-1]
    #             #ax.axvspan(t1, t2, facecolor='g', alpha=0.2)
    #             ax2.hlines(y=1, xmin=t1*1e3, xmax=t2*1e3,
    #                        linewidth=8, color='k', alpha=0.3)
    #             sig_period = False
    #         elif sig_period and (i_t==len(reject_fdr)-1): # Last time point
    #             t2 = times[i_t]
    #             ax2.hlines(y=1, xmin=t1*1e3, xmax=t2*1e3,
    #                        linewidth=8, color='k', alpha=0.3)

    
    feature_names = feature_info.keys()
    for i_feature, feature_name in enumerate(feature_names):
        
        color, ls, lw, marker = get_curve_style(feature_name, feature_info)
        st, ed = feature_info[feature_name]['IXs']
        if group:
            # IX_max_abs = np.argmax(np.abs(coefs_mean[st:ed, :]), axis=0)
            coefs = coefs_mean['full'][:, st:ed]
            n_coefs = coefs.shape[1]
            coef_curr_feature_group_mean = np.mean(coefs, axis=1)
            coef_curr_feature_group_sem = np.std(coefs, axis=1)/np.sqrt(n_coefs)
            ax.plot(times, coef_curr_feature_group_mean,
                        color=color, ls=ls, lw=lw,
                    marker=marker, markersize=25, label=feature_name)
            ax.fill_between(x=times,
                            y1=coef_curr_feature_group_mean-coef_curr_feature_group_sem,
                            y2=coef_curr_feature_group_mean+coef_curr_feature_group_sem,
                            alpha=0.2,
                            color=color)
        else:
       #     for i_value, feature_value in enumerate(feature_info[feature_name]['names']):
            coef_curr_feature = coefs_mean['full'][:, st:ed]
            #ax.plot(times, coef_curr_feature, color=color, ls=ls, lw=lw, label=feature_info[feature_name]['names'])
            ax.plot(times, coef_curr_feature, ls=ls, lw=lw, label=feature_info[feature_name]['names'])
            print(feature_name)
    
    #ax.legend(loc='center left', bbox_to_anchor=(1.5, 0, 0.5, 1.2), ncol=int(np.ceil(len(feature_names)/10)), fontsize=16)
    ax.legend(loc='center left', bbox_to_anchor=(1.2, 0, 0.5, 1), ncol=int(np.ceil(coefs_mean['full'].shape[1]/30)), fontsize=16)
    ax.set_xlabel('Time (msec)', fontsize=30)
    ax.set_ylabel(r'Beta', fontsize=30)
    ax.set_ylim((None, None)) 
    if args.block_type == 'visual':
        ax.axvline(x=0, ls='--', color='k')
        ax.axvline(x=500, ls='--', color='k')
    ax.axhline(ls='--', color='k')    
    # ax.set_xlim((-0.250, 0.750))
    ax.tick_params(axis='both', labelsize=18)
    #ax2.tick_params(axis='both', labelsize=18)
    plt.subplots_adjust(right=0.45)
    
    return fig


def plot_evoked_bar_r(times, scores_mean, scores_sem, reject_fdr,
              ch_name, feature_info, args, y_lim=None, keep=False):
    fig, ax = plt.subplots(figsize=(13,10))
    
    # Draw full-model results
    # scores_full_mean = scores['full']['scores_by_time'][0][i_channel, :]
    # print(scores_full_mean)
    # print(scores_full_mean.shape)
    color = 'k'
    # ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_ylabel('Correlation coefficient ($r$)', color=color, fontsize=40)
    # ax2.plot(times*1e3, scores_mean['full'], color=color, lw=3)
    # ax2.fill_between(times*1e3,
    #                 scores_mean['full'] + scores_sem['full'],
    #                 scores_mean['full'] - scores_sem['full'],
    #                 color=color,
    #                 alpha=0.2)
    # ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_xlim((-250, 750))
    
    if any(reject_fdr['full']):
        sig_period = False
        for i_t, reject in enumerate(reject_fdr['full']):
            if reject and (not sig_period): # Entering a significance zone
                t1 = times[i_t]
                sig_period = True
            elif (not reject) and sig_period: # Exiting a sig zone
                t2 = times[i_t-1]
                #ax.axvspan(t1, t2, facecolor='g', alpha=0.2)
                # ax2.hlines(y=1, xmin=t1*1e3, xmax=t2*1e3,
                #            linewidth=8, color='k', alpha=0.3)
                sig_period = False
            elif sig_period and (i_t==len(reject_fdr)-1): # Last time point
                t2 = times[i_t]
                # ax2.hlines(y=1, xmin=t1*1e3, xmax=t2*1e3,
                #            linewidth=8, color='k', alpha=0.3)

    feature_names, colors = [], []  # performance of the full model must be calculated
    if args.each_feature_value:
        for f in feature_info.keys():
            for f_name in feature_info[f]['names']:
                feature_names.append(f_name)
    else:
        feature_names = feature_info.keys()
    
    n_features = len(feature_names)

    if y_lim is None: 
        y_lim = 1 + (1+n_features)*0.02
        # ax2.set_ylim((0, y_lim))
        y_lim = 0.05 +(1+n_features)*0.02
    # else:
    #     ax2.set_ylim((0, 0.6))
    feature_importance_mean, feature_importance_max, feature_importance_sem = [], [], []
    for i_feature, feature_name in enumerate(feature_names):
        color, ls, lw, marker = get_curve_style(feature_name, feature_info)
        mask_sig = np.zeros_like(reject_fdr[feature_name])
        if any(reject_fdr[feature_name]):
            sig_period = False
            for i_t, reject in enumerate(reject_fdr[feature_name]):
                if reject and (not sig_period): # Entering a significance zone
                    t1 = times[i_t]
                    sig_period = True
                    i_t_st = i_t
                elif (not reject) and sig_period: # Exiting a sig zone
                    t2 = times[i_t-1]
                    i_t_ed = i_t
                    #ax.axvspan(t1, t2, facecolor='g', alpha=0.2)
                    # ax.hlines(y=1.02+i_feature*0.02, xmin=t1*1e3, xmax=t2*1e3,
                    #            linewidth=8, color=color, alpha=0.3)
                    sig_period = False
                    mask_sig[i_t_st:i_t_ed] = 1
                elif sig_period and (i_t==len(reject_fdr)-1): # Last time point
                    t2 = times[i_t]
                    i_t_ed = -1
                    # ax.hlines(y=y_lim+i_feature*0.02, xmin=t1*1e3, xmax=t2*1e3,
                    #            linewidth=8, color=color, alpha=0.3)
                    mask_sig[i_t_st:i_t_ed] = 1

        
        
        if keep:
            # feature_importance = scores_mean[feature_name]*(2*scores_mean[feature_name])/(scores_mean['full'] + scores_mean[feature_name])
            feature_importance = scores_mean[feature_name]
        else:
            #feature_importance = scores_mean[feature_name]*(scores_mean['full'] - scores_mean[feature_name])/(scores_mean['full'] + scores_mean[feature_name])
            feature_importance = (scores_mean['full'] - scores_mean[feature_name])
        feature_importance = np.maximum(feature_importance, np.zeros_like(feature_importance))
        
        IXs = np.where(np.logical_and(times<=0.6, times>=0))
        feature_importance_mean.append(feature_importance[IXs].mean())
        feature_importance_max.append(feature_importance[IXs].max())
        feature_importance_sem.append(feature_importance[IXs].std()/np.sqrt(feature_importance[IXs].size))
        colors.append(color)
        # feature_importance = np.minimum(feature_importance, 2*np.ones_like(feature_importance))
        # feature_importance[~mask_sig] = 0
        #print(feature_name, color, ls, lw, marker, feature_importance)
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, feature_importance_mean, xerr=feature_importance_sem, color=colors)#, align='center')
    ax.set_yticks(y_pos, labels=feature_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    
    # ax.set_xlabel('Time (msec)', fontsize=40)
    # ax.set_ylabel(r'$\Delta r$', fontsize=40)
    ax.set_xlim((0, 1.5*np.max(feature_importance_mean)))
    ax.set_xticks([0, np.max(feature_importance_mean)])
    ax.set_xlabel('Feature importance', fontsize=40)
    #ax.set_ylim((0, y_lim+(1+n_features)*0.02))
    # ax.set_ylim((0, y_lim))
    # if args.block_type == 'visual':
    #     ax.axvline(x=0, ls='--', color='k')
    #     ax.axvline(x=500, ls='--', color='k')
    # ax.axhline(ls='--', color='k')
    
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', labelsize=35)
    # ax2.tick_params(axis='both', labelsize=35)
    # ax.legend(loc='center left', bbox_to_anchor=(1.5, 0, 0.5, 1.2), ncol=int(np.ceil(len(feature_names)/20)), fontsize=16)
    plt.subplots_adjust(left=0.25)

    return fig



def plot_evoked_bar_coef(times, coef_mean, coef_sem, reject_fdr,
              ch_name, feature_info, args, y_lim=None, keep=False):
    fig, ax = plt.subplots(figsize=(13,10))
    
    # Draw full-model results
    # scores_full_mean = scores['full']['scores_by_time'][0][i_channel, :]
    # print(scores_full_mean)
    # print(scores_full_mean.shape)
    color = 'k'
    # ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_ylabel('Correlation coefficient ($r$)', color=color, fontsize=40)
    # ax2.plot(times*1e3, scores_mean['full'], color=color, lw=3)
    # ax2.fill_between(times*1e3,
    #                 scores_mean['full'] + scores_sem['full'],
    #                 scores_mean['full'] - scores_sem['full'],
    #                 color=color,
    #                 alpha=0.2)
    # ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_xlim((-250, 750))
    
    if any(reject_fdr['full']):
        sig_period = False
        for i_t, reject in enumerate(reject_fdr['full']):
            if reject and (not sig_period): # Entering a significance zone
                t1 = times[i_t]
                sig_period = True
            elif (not reject) and sig_period: # Exiting a sig zone
                t2 = times[i_t-1]
                #ax.axvspan(t1, t2, facecolor='g', alpha=0.2)
                # ax2.hlines(y=1, xmin=t1*1e3, xmax=t2*1e3,
                #            linewidth=8, color='k', alpha=0.3)
                sig_period = False
            elif sig_period and (i_t==len(reject_fdr)-1): # Last time point
                t2 = times[i_t]
                # ax2.hlines(y=1, xmin=t1*1e3, xmax=t2*1e3,
                #            linewidth=8, color='k', alpha=0.3)

    feature_names, colors = [], []  # performance of the full model must be calculated
    if args.each_feature_value:
        for f in feature_info.keys():
            for f_name in feature_info[f]['names']:
                feature_names.append(f_name)
    else:
        feature_names = feature_info.keys()
    
    n_features = len(feature_names)

    if y_lim is None: 
        y_lim = 1 + (1+n_features)*0.02
        # ax2.set_ylim((0, y_lim))
        y_lim = 0.05 +(1+n_features)*0.02
    # else:
    #     ax2.set_ylim((0, 0.6))
    feature_importance_mean, feature_importance_max, feature_importance_sem = [], [], []
    for i_feature, feature_name in enumerate(feature_names):
        color, ls, lw, marker = get_curve_style(feature_name, feature_info)
        mask_sig = np.zeros_like(reject_fdr[feature_name])
        if any(reject_fdr[feature_name]):
            sig_period = False
            for i_t, reject in enumerate(reject_fdr[feature_name]):
                if reject and (not sig_period): # Entering a significance zone
                    t1 = times[i_t]
                    sig_period = True
                    i_t_st = i_t
                elif (not reject) and sig_period: # Exiting a sig zone
                    t2 = times[i_t-1]
                    i_t_ed = i_t
                    #ax.axvspan(t1, t2, facecolor='g', alpha=0.2)
                    ax.hlines(y=1.02+i_feature*0.02, xmin=t1*1e3, xmax=t2*1e3,
                               linewidth=8, color=color, alpha=0.3)
                    sig_period = False
                    mask_sig[i_t_st:i_t_ed] = 1
                elif sig_period and (i_t==len(reject_fdr)-1): # Last time point
                    t2 = times[i_t]
                    i_t_ed = -1
                    ax.hlines(y=y_lim+i_feature*0.02, xmin=t1*1e3, xmax=t2*1e3,
                               linewidth=8, color=color, alpha=0.3)
                    mask_sig[i_t_st:i_t_ed] = 1

        
        
        feature_importance = coef_mean[feature_name]
        
        #feature_importance = np.maximum(feature_importance, np.zeros_like(feature_importance))
        
        IXs = np.where(np.logical_and(times<=0.6, times>=0))
        feature_importance_mean.append(feature_importance[IXs].mean())
        feature_importance_max.append(feature_importance[IXs].max())
        feature_importance_sem.append(feature_importance[IXs].std()/np.sqrt(feature_importance[IXs].size))
        colors.append(color)
        # feature_importance = np.minimum(feature_importance, 2*np.ones_like(feature_importance))
        # feature_importance[~mask_sig] = 0
        #print(feature_name, color, ls, lw, marker, feature_importance)
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, feature_importance_mean, xerr=feature_importance_sem, color=colors)#, align='center')
    ax.set_yticks(y_pos, labels=feature_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    
    # ax.set_xlabel('Time (msec)', fontsize=40)
    # ax.set_ylabel(r'$\Delta r$', fontsize=40)
    ax.set_xlim((0, 1.5*np.max(feature_importance_mean)))
    ax.set_xticks([0, np.max(feature_importance_mean)])
    ax.set_xlabel('Feature importance', fontsize=40)
    #ax.set_ylim((0, y_lim+(1+n_features)*0.02))
    # ax.set_ylim((0, y_lim))
    # if args.block_type == 'visual':
    #     ax.axvline(x=0, ls='--', color='k')
    #     ax.axvline(x=500, ls='--', color='k')
    # ax.axhline(ls='--', color='k')
    
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', labelsize=35)
    # ax2.tick_params(axis='both', labelsize=35)
    # ax.legend(loc='center left', bbox_to_anchor=(1.5, 0, 0.5, 1.2), ncol=int(np.ceil(len(feature_names)/20)), fontsize=16)
    plt.subplots_adjust(left=0.25)

    return fig
