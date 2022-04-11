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
from viz import plot_rf_coefs, plot_rf_r2
sys.path.append('..')
from utils.utils import dict2filename
import matplotlib.pyplot as plt
from encoding.models import TimeDelayingRidgeCV
import numpy as np
import pandas as pd
import seaborn as sns
from mne.stats import fdr_correction

parser = argparse.ArgumentParser(description='Plot TRF results')
# DATA
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike', 'microphone'],
                    default='spike', help='electrode type')
parser.add_argument('--filter', default='raw', help='')
parser.add_argument('--smooth', default=50,
                    help='Gaussian smoothing in msec')
parser.add_argument('--probe-name', default=None, nargs='*',
                    action='append', type=str,
                    help='Probe name to plot (ignores channel-name/num)')
parser.add_argument('--channel-name', default=[], nargs='*', action='append',
                    type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=[], nargs='*', action='append',
                    type=int, help='channel number (if empty all channels)')
# MISC
parser.add_argument('--path2output',
                    default=os.path.join('..', '..', '..',
                                         'Output', 'encoding_models'))
parser.add_argument('--path2figures',
                    default=os.path.join('..', '..', '..',
                                         'Figures', 'encoding_models', 'scatters'))
parser.add_argument('--decimate', default=50, type=float,
                    help='If not empty, decimate data for speed.')
parser.add_argument('--model-type', default='ridge',
                    choices=['ridge', 'lasso', 'ridge_laplacian', 'standard'])
parser.add_argument('--ablation-method', default='remove',
                    choices=['shuffle', 'remove', 'zero'],
                    help='Method used to calcuated feature importance\
                        by reducing/ablating a feature family')
#parser.add_argument('--query-train', default="block in [2,4,6] and word_length>1")
#parser.add_argument('--query-test', default="block in [2,4,6] and word_length>1")
parser.add_argument('--each-feature-value', default=False, action='store_true',
                    help="Evaluate model after ablating each feature value. \
                         If false, ablate all feature values together")


#############
# USER ARGS #
#############

alpha = 0.05

args = parser.parse_args()
fn_trf_results = f'../../../Output/encoding_models/evoked_encoding_results_decimate_{args.decimate}_smooth_{args.smooth}.json'
df = pd.read_json(fn_trf_results)
df = df.loc[df['data_type'] == args.data_type]
df = df.loc[df['filter'] == args.filter]
df = df.loc[df['Feature']=='full']


for block in ['auditory', 'visual']:
    pvals = np.asarray([np.asarray(a) for a in df[f'stats_full_{block}_by_time'].values])
    pvals_cat = np.concatenate(pvals)
    reject_fdr, pvals_fdr = fdr_correction(pvals_cat,
                                           alpha=alpha,
                                           method='indep')
    df[f'pvals_full_{block}_fdr_spatiotemporal'] = pvals_fdr.reshape((pvals.shape[0], -1)).tolist()

def get_r_significant(row, block, alpha):
    rs = row[f'r_full_{block}_by_time']
    mask = np.logical_and(np.asarray(row[f'pvals_full_{block}_fdr_spatiotemporal']) <= alpha,
                          np.asarray(row[f'r_full_{block}_by_time']) > 0)
    if mask is None:
        return None
    rs_masked = np.asarray(rs)[mask]
    return rs_masked



for block in ['auditory', 'visual']:
    df[f'r_significant_full_{block}'] = \
        df.apply(lambda row: get_r_significant(row, block, alpha),
                 axis=1)

def mean_r(row, block):
    if row[f'r_significant_full_{block}'].size>0:
        return np.asarray(row[f'r_significant_full_{block}']).mean()
    else:
        return None

for block in ['auditory', 'visual']:
    df[f'r_mean_significant_full_{block}'] = \
        df.apply(lambda row:  mean_r(row, block),
                 axis=1)


def get_probe_name(row):
    if row['data_type'] == 'micro':
        probe_name = None
    elif row['data_type'] == 'macro':
        probe_name = None
    elif row['data_type'] == 'spike':
        if row['Ch_name'].startswith('G'):
            probe_name = ''.join([c for c in row['Ch_name'].split('-')[1].split('_')[0] if not c.isdigit()])[1:]
        else:
            probe_name = None
    else:
        print(row)
        raise('Wrong data type')
    return probe_name

def get_hemisphere(row):
    if row['data_type'] == 'micro':
        probe_name = None
    elif row['data_type'] == 'macro':
        probe_name = None
    elif row['data_type'] == 'spike':
        if row['Ch_name'].startswith('G'):
            probe_name = ''.join([c for c in row['Ch_name'].split('-')[1].split('_')[0] if not c.isdigit()])[0]
        else:
            probe_name = None
    else:
        print(row)
        raise('Wrong data type')
    return probe_name

df['probe_name'] = df.apply(lambda row: get_probe_name(row), axis=1)
df['hemisphere'] = df.apply(lambda row: get_hemisphere(row), axis=1)

# df.drop(columns='Hemisphere')
# df = df.dropna()

def get_ROIs(row):
    if row['probe_name'] is not None:
        if ('FG' in row['probe_name']):
            return 'Brodmann-37'
        if ('FSG' in row['probe_name']):
            return 'Brodmann-37'
        if ('STG' in row['probe_name']):
            return 'Brodmann-22'
        elif ('MTG' in row['probe_name']):
            return 'Brodmann-21'
        else:
            return ''
    else:
        return ''
            
df['ROI'] = df.apply(lambda row:get_ROIs(row), axis=1)
df = df.loc[df['ROI'] != '']

print(df)

fig_scatter, ax_scatter = plt.subplots(figsize=(10, 10))
sns.scatterplot(data=df,
                x="r_mean_significant_full_auditory",
                y="r_mean_significant_full_visual",
                hue="ROI",
                style="hemisphere",
                s=100)

# COSMETICS
lims = (0.05, 0.2)
ax_scatter.set_xlim(lims)
ax_scatter.set_ylim(lims)
ax_scatter.set_aspect('equal', adjustable='box')
ax_scatter.set_xlabel('Brain Score (Auditory)', fontsize=20)
ax_scatter.set_ylabel('Brain Score (Visual)', fontsize=20)
ax_scatter.tick_params(axis='both', labelsize=20)
ax_scatter.plot([0, 1], [0, 1], transform=ax_scatter.transAxes, ls='--', color='k', lw=2)
# ax_scatter.legend().set_visible(False)
fn = f'scatter_{args.data_type}_{args.filter}.png'
fig_scatter.savefig(os.path.join(args.path2figures, fn))
print(f'saved to: {args.path2figures}/{fn}')
plt.close(fig_scatter) 


# # append rows to an empty DataFrame
# fig2, ax_bar = plt.subplots(figsize=(10, 10))
# ax_bar = sns.barplot(x="probe_name", y="r_mean_significant_full_auditory", data=df)
# fn = f'bar_{args.data_type[0]}_{args.filter[0]}.png'
# fig2.savefig(os.path.join(args.path2figures, fn))


#print(dict_scatter)
# if not os.path.exists(args.path2figures):
#     os.makedirs(args.path2figures)

# print('Plotting...')
# for probe_name in dict_scatter.keys():
#     fig1, ax_scatter = plt.subplots(figsize=(10, 10))
#     for feature in list(feature_list) + ['full']:
#         ch_names = dict_scatter[probe_name][feature].keys()
#         n_points = len(ch_names)
#         Xs, Ys = [], []
#         for ch_name in ch_names:
#             if feature == 'phonology':
#                 X = dict_scatter[probe_name][feature][ch_name]['auditory']
#                 Y = dict_scatter[probe_name]['full'][ch_name]['visual'] # Y=0 after subtraction of full below
#             elif feature == 'orthography':
#                 X = dict_scatter[probe_name]['full'][ch_name]['auditory']
#                 Y = dict_scatter[probe_name][feature][ch_name]['visual']
#             else:
#                 X = dict_scatter[probe_name][feature][ch_name]['auditory']
#                 Y = dict_scatter[probe_name][feature][ch_name]['visual']
            
#             if feature != 'full':
#                 X = dict_scatter[probe_name]['full'][ch_name]['auditory'] - X
#                 Y = dict_scatter[probe_name]['full'][ch_name]['visual'] - Y
#                 color = feature_info[feature]['color']
#             else:
#                 color = 'k'
#             Xs.append(X)
#             Ys.append(Y)
#         if feature == 'semantics':
#             color = 'xkcd:orange' 
#         #print(Xs, Ys, color)
#         ax_scatter.scatter(Xs, Ys, color=color)

#     ax_scatter.set_xlabel('Auditory', fontsize=16)
#     ax_scatter.set_ylabel('Visual', fontsize=16)
#     ax_scatter.set_xlim([-0.5, 1])
#     ax_scatter.set_ylim([-0.5, 1])
#     ax_scatter.plot([0, 1], [0, 1], transform=ax_scatter.transAxes, ls='--', color='k', lw=2)
#     plt.title(f'{probe_name} ({args.data_type[0]}, {args.filter[0]})', fontsize=20)
#     fn = f'scatter_{probe_name}_{args.data_type[0]}_{args.filter[0]}.png'
#     fig.savefig(os.path.join(args.path2figures, fn))
#     print(f'saved to: {args.path2figures}/{fn}')
#     plt.close(fig) 
