#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:48:59 2021

@author: yl254115
"""
import argparse
import os
import sys
import pickle
from viz import plot_rf_coefs, plot_rf_r2
sys.path.append('..')
from utils.utils import dict2filename
import matplotlib.pyplot as plt
from encoding.models import TimeDelayingRidgeCV
import pandas as pd
import numpy as np
import seaborn as sns

parser = argparse.ArgumentParser(description='Plot TRF results')
# DATA
parser.add_argument('--patient', action='append', default=['515'],
                    help='Patient string')
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike'],
                    action='append', default=['micro'], help='electrode type')
parser.add_argument('--filter', action='append',
                    default=['raw'],
                    help='raw/high-gamma')
parser.add_argument('--smooth', default=25,
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
                                         'Figures', 'encoding_models'))
parser.add_argument('--decimate', default=None, type=float,
                    help='If not empty, decimate data for speed.')
parser.add_argument('--model-type', default='ridge',
                    choices=['ridge', 'lasso', 'ridge_laplacian', 'standard'])
parser.add_argument('--ablation-method', default='remove',
                    choices=['shuffle', 'remove', 'zero'],
                    help='Method used to calcuated feature importance\
                        by reducing/ablating a feature family')
parser.add_argument('--query_train', default="block in [2,4,6] and word_length>1")
parser.add_argument('--query_test', default="block in [2,4,6] and word_length>1")
parser.add_argument('--each-feature-value', default=False, action='store_true',
                    help="Evaluate model after ablating each feature value. \
                         If false, ablate all feature values together")


#############
# USER ARGS #
#############
args = parser.parse_args()
assert len(args.patient) == len(args.data_type) == len(args.filter)
args.patient = ['patient_' + p for p in args.patient]
args.block_type = 'both'
if not args.query_test:
    args.query_test = args.query_train
print('args\n', args)
list_args2fname = ['patient', 'data_type', 'filter', 'smooth',
                   'model_type', 'ROI', 'ablation_method',
                   'query_train', 'each_feature_value']
if args.query_train != args.query_test:
    list_args2fname.extend(['query_test'])

if not os.path.exists(args.path2figures):
    os.makedirs(args.path2figures)

#########################
args2fname = args.__dict__.copy()
fname = dict2filename(args2fname, '_', list_args2fname, '', True)

#########################
# LOAD ENCODING RESULTS #
#########################

df = pd.read_csv(f'../../../Output/encoding_models/trf_results_{args.data_type[0]}_{args.filter[0]}.csv')

def get_dr(r_full, r):
    dr = r_full - r
    # if r_full > 0.05:
    #     dr = r_full-r
    # else:
    #     dr = 0
    return dr


def lump_ROIs(row):
    
    MTL=['AH', 'A', 'MH', 'EC']
    primary_auditory = ['HGa', 'HSG']
    occipital=['O', 'IO', 'TO']
    Fusiform = ['FGP', 'FSG', 'FGA']
    frontal = ['OF', 'AF', 'IF', 'FOP', 'OPR', 'OBI', 'OF-AC', 'IFAC']
    SMG = ['SM', 'PSM', 'SMG', 'PI-SMGa']
    STG = ['STG', 'PSTG', 'ASTG']
    MTG = ['MTG', 'PMTG']
    cingulate = ['AC', 'PC']
    insula = ['AI', 'MI']
    
    if row['ROI'] in MTL:
        ROI_large = 'MTL'
    elif row['ROI'] in primary_auditory:
        ROI_large = 'Primary Auditory'
    elif row['ROI'] in occipital:
        ROI_large = 'Occipital'
    elif row['ROI'] in Fusiform:
        ROI_large = 'Fusiform'
    elif row['ROI'] in frontal:
        ROI_large = 'Frontal'
    elif row['ROI'] in SMG:
        ROI_large = 'SMG'
    elif row['ROI'] in STG:
        ROI_large = 'STG'
    elif row['ROI'] in MTG:
        ROI_large = 'MTG'
    elif row['ROI'] in cingulate:
        ROI_large = 'Cingulate'
    elif row['ROI'] in insula:
        ROI_large = 'Insula'
    else:
        ROI_large = row['ROI']
    return ROI_large


df['dr_visual'] = df.apply (lambda row: get_dr(row['r_full_visual'],
                                               row['r_visual']), axis=1)
df['dr_auditory'] = df.apply (lambda row: get_dr(row['r_full_auditory'],
                                                 row['r_auditory']), axis=1)
df['ROI'] = df.apply (lambda row: row['Probe_name'][1:], axis=1)

df['ROI_large'] = df.apply(lambda row: lump_ROIs(row), axis=1)

print(df)

################
palette ={'semantics':'orange',
          'is_last_word':'k',
          'lexicon':'g',
          'is_first_word':'grey',
          'syntax':'b',
          'phonology':'m',
          'orthography':'r',
          'full':'k'}
#################


dict_filter = {'raw':'Broadband', 'high-gamma':'High-gamma'}


df = df[~df['Ch_name'].str.startswith('C')]

##### SCATTER FULL ALL PROBES
###############################

df_full = df[df['Feature'] == 'full']

sns.set(font_scale=2)
# VISUAL VS AUDITORY BLOCKS
fig, ax = plt.subplots(figsize=(20, 10))
ax = sns.scatterplot(data=df_full,
                     x="r_visual",
                     y="r_auditory",
                     hue='ROI_large',
                     style="Hemisphere",
                     ax=ax)
ax.set_xlim([-0.05, 1])
ax.set_ylim([-0.05, 1])
ax.set_title(f'Fit score (corrcoef; {args.data_type[0]}, {dict_filter[args.filter[0]]})')
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--', color='k', lw=2)
ax.legend(bbox_to_anchor=(1.02, 0, 0.3, 1), loc='center left', ncol=2)
ax.set_xlabel('Visual')
ax.set_ylabel('Auditory')
plt.gca().set_aspect('equal', adjustable='box')
plt.subplots_adjust(right=0.6)
fn = f'../../../Figures/encoding_models/scatters/scatter_{args.data_type[0]}_{args.filter[0]}.png'
fig.savefig(fn)
plt.close(fig)
print(fn)

######## FIGURES PER PROBE
##############################
ROIs = list(set(df_full['ROI_large']))
for ROI in ROIs:
    
    # SCATTER FULL PER PROBE
    df_full_probe = df_full[df_full['ROI_large'] == ROI]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.scatterplot(data=df_full_probe,
                         x="r_visual",
                         y="r_auditory",
                         style='Hemisphere',
                         ax=ax)
    ax.set_xlim([-0.05, 1])
    ax.set_ylim([-0.05, 1])
    ax.set_title(f'{ROI} ({args.data_type[0]}, {args.filter[0]})')
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--', color='k', lw=2)
    fn = f'../../../Figures/encoding_models/scatters/scatter_full_{ROI}_{args.data_type[0]}_{args.filter[0]}.png'
    fig.savefig(fn)
    plt.close(fig)
    print(f'Figure saved to: {fn}')
    
    # BAR PLOT PER PROBE (for both AUD and VIS blocks)
    df_probe = df[(df['ROI_large'] == ROI) & (df['Feature'] != 'full')]
    
    # fig, ax = plt.subplots(2, 1, figsize=(15,10))
    g = sns.catplot(data=df_probe,
                    x="ROI_large",
                    y="dr_auditory",
                    hue='Feature',
                    col='Hemisphere',
                    palette=palette,
                    # ci=68,
                    kind='bar',
                    ax=ax)
    # l = axs[0].get_legend()
    # l.remove()
    # axs[0].set(ylim=(0, None))
    # axs[0].set_title(f'Auditory ({args.data_type[0]}, {args.filter[0]})')
    # plt.subplots_adjust(right=0.75)
    # fig = ax.get_figure()
    fn = f'../../../Figures/encoding_models/scatters/bar_{ROI}_aud_{args.data_type[0]}_{args.filter[0]}.png'
    # plt.show()
    g.savefig(fn)
    # plt.close(fig)
    
    g = sns.catplot(data=df_probe,
                    x="ROI_large",
                    y="dr_visual",
                    hue='Feature',
                    col='Hemisphere',
                    palette=palette,
                    # capsize=.05,
                    # ci=68,
                    kind='bar')
    # axs[1].set(ylim=(0, None))
    # axs[1].set_title(f'Visual ({args.data_type[0]}, {args.filter[0]})')
    # axs[1].legend(loc='center left', bbox_to_anchor=(1.05, 0, 0.3, 1))
    # plt.subplots_adjust(right=0.75)
    # fig = plt.get_figure()
    fn = f'../../../Figures/encoding_models/scatters/bar_{ROI}_vis_{args.data_type[0]}_{args.filter[0]}.png'
    g.savefig(fn)
    plt.close(fig)
    print(f'Figure saved to: {fn}')
    
    # SCATTER PER FETAURE PER PROBE (without full)
    fig, ax = plt.subplots(figsize=(20, 10))
    ax = sns.scatterplot(data=df_probe,
                         x="dr_visual",
                         y="dr_auditory",
                         hue="Feature",
                         style='Hemisphere',
                         palette=palette,
                         ax=ax,
                         alpha=0.5)
    ax.set_xlim([-0.05, 0.1])
    ax.set_ylim([-0.05, 0.1])
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--', color='k', lw=2)
    ax.legend(bbox_to_anchor=(1.02, 0, 0.3, 1), loc='center left')
    plt.subplots_adjust(right=0.7)
    
    fn = f'../../../Figures/encoding_models/scatters/scatter_feature_{ROI}_{args.data_type[0]}_{args.filter[0]}.png'
    fig.savefig(fn)
    plt.close(fig)
    print(f'Figure saved to: {fn}')
    
###########

# SCATTER PER FEATURE 
df = df[df['Feature'] != 'full']
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.scatterplot(data=df,
                     x="dr_visual",
                     y="dr_auditory",
                     hue="Feature",
                     style='Hemisphere',
                     palette=palette,
                     ax=ax,
                     alpha=0.5)
ax.set_xlim([-0.05, 0.1])
ax.set_ylim([-0.05, 0.1])
ax.legend(bbox_to_anchor=(1.02, 0, 0.3, 1), loc='center left')


# fig, ax = plt.subplots(figsize=(50, 10))
# ax = sns.catplot(data=df,
#                  x="ROI",
#                  y="dr_visual",
#                  hue='Feature',
#                  row="Hemisphere",
#                  kind="bar",
#                  palette=palette,
#                  ax=ax)

# ax.set(ylim=(0, None))
# ax._legend.remove()

# fig, ax = plt.subplots(figsize=(50, 10))
# ax = sns.catplot(data=df,
#                  x="ROI",
#                  y="dr_auditory",
#                  hue='Feature',
#                  row="Hemisphere",
#                  kind="bar",
#                  palette=palette,
#                  ax=ax)

# ax.set(ylim=(0, None))
