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
import ast

parser = argparse.ArgumentParser(description='Plot TRF results')
# DATA
parser.add_argument('--patient', action='append', default=[],
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
#assert len(args.patient) == len(args.data_type) == len(args.filter)
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
    if isinstance(r_full, float):
        if np.isnan(r_full) or np.isnan(r):
            return np.nan
        else:
            dr = r_full - r
    else:
        dr = r_full - r
    # if r_full > 0.05:
    #     dr = r_full-r
    # else:
    #     dr = 0
    return dr


def str2list(s):
    if isinstance(s, str):
        val_list = s[1:-1].split()
        val_list = np.asarray([float(v) for v in val_list])
    else:
        val_list = s
    return val_list


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
    
    if args.data_type[0] == 'spike':
        if row['ROI'].startswith('one'):  # HACK
            ROI = 'None'
        else:
            ROI = row['ROI'].split('_')[0].split('-')[1][1:-1]
            
    else:
        ROI = row['ROI']
    
    if ROI in MTL:
        ROI_large = 'MTL'
    elif ROI in primary_auditory:
        ROI_large = 'Primary Auditory'
    elif ROI in occipital:
        ROI_large = 'Occipital'
    elif ROI in Fusiform:
        ROI_large = 'Fusiform'
    elif ROI in frontal:
        ROI_large = 'Frontal'
    elif ROI in SMG:
        ROI_large = 'SMG'
    elif ROI in STG:
        ROI_large = 'STG'
    elif ROI in MTG:
        ROI_large = 'MTG'
    elif ROI in cingulate:
        ROI_large = 'Cingulate'
    elif ROI in insula:
        ROI_large = 'Insula'
    else:
        ROI_large = ROI
    return ROI_large

def probename2ROI(path2mapping='../../../Data/probenames2fsaverage.tsv'):
    with open(path2mapping, 'r') as f:
        lines = f.readlines()
    atlas_regions = [line.split('\t')[0] for line in lines if line.split('\t')[1].strip('\n')!='-']
    probe_names = [line.split('\t')[1].strip('\n') for line in lines if line.split('\t')[1].strip('\n')!='-']
    p2r = {}
    for atlas_region, probe_name in zip(atlas_regions, probe_names):
        for probename in probe_name.split(','): # could be several, comma delimited
            assert probename not in p2r.keys()
            p2r[probename.strip()] = atlas_region
    return p2r

probename2roi = probename2ROI()

df['r_full_visual_by_time'] = df.apply (lambda row: str2list(row['r_full_visual_by_time']), axis=1)
df['r_full_auditory_by_time'] = df.apply (lambda row: str2list(row['r_full_auditory_by_time']), axis=1)
df['r_visual_by_time'] = df.apply (lambda row: str2list(row['r_visual_by_time']), axis=1)
df['r_auditory_by_time'] = df.apply (lambda row: str2list(row['r_auditory_by_time']), axis=1)

df['dr_visual_total'] = df.apply (lambda row: get_dr(row['r_full_visual'],
                                               row['r_visual']), axis=1)
df['dr_auditory_total'] = df.apply (lambda row: get_dr(row['r_full_auditory'],
                                                 row['r_auditory']), axis=1)
df['dr_visual_max'] = df.apply (lambda row: np.max(get_dr(row['r_full_visual_by_time'],
                                               row['r_visual_by_time'])), axis=1)
df['dr_auditory_max'] = df.apply (lambda row: np.max(get_dr(row['r_full_auditory_by_time'],
                                                 row['r_auditory_by_time'])), axis=1)

df['ROI'] = df.apply (lambda row: row['Probe_name'][1:], axis=1)

def get_ROI(Probe_name):
    if Probe_name.isdigit():
        return None
    else:
        return probename2roi[Probe_name]

df['ROI_large'] = df.apply(lambda row: get_ROI(row['Probe_name']), axis=1)

print(df)

################
palette ={'word_onset':'y',
          'positional':'y',
          'semantics':'orange',
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


# Summary point plot for visual and auditory block with ALL ROIs

fig, axs = plt.subplots(1, 2, figsize=(30, 30))
for i_mod, modality in enumerate(['visual', 'auditory']):
    df_features = df[df['Feature'] != 'full']
    g = sns.pointplot(data=df_features,
                      y="ROI_large",
                      x=f"dr_{modality}_total",
                      hue='Feature',
                      join=False,
                      palette=palette,
                      ax=axs[i_mod])
    if i_mod == 0:
        g.get_legend().remove()
    if i_mod == 1:
        g.set(yticklabels=[], ylabel=None)
    g.set(xlim=(0, None))
axs[1].legend(bbox_to_anchor=(1.02, 0, 0.3, 1), loc='center left')
plt.subplots_adjust(right=0.85)
fn = f'../../../Figures/encoding_models/scatters/All_ROIs_{args.data_type[0]}_{args.filter[0]}.png'
fig.savefig(fn)
plt.close(fig)
print(fn)


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
                    y="dr_auditory_total",
                    hue='Feature',
                    col='Hemisphere',
                    palette=palette,
                    # ci=68,
                    kind='bar',
                    ax=ax)
    g.set(ylim=[0,None],xlabel='', ylabel='Feature importance', title='', yticklabels=[])
    fn = f'../../../Figures/encoding_models/scatters/bar_{ROI}_aud_{args.data_type[0]}_{args.filter[0]}.png'
    g.savefig(fn)
    
    g = sns.catplot(data=df_probe,
                    x="ROI_large",
                    y="dr_visual_total",
                    hue='Feature',
                    col='Hemisphere',
                    palette=palette,
                    # capsize=.05,
                    # ci=68,
                    kind='bar')
    g.set(ylim=[0,None],xlabel='', ylabel='Feature importance', title='', yticklabels=[])
    fn = f'../../../Figures/encoding_models/scatters/bar_{ROI}_vis_{args.data_type[0]}_{args.filter[0]}.png'
    g.savefig(fn)
    plt.close(fig)
    print(f'Figure saved to: {fn}')
    
    # SCATTER PER FETAURE PER PROBE (without full)
    fig, ax = plt.subplots(figsize=(20, 10))
    ax = sns.scatterplot(data=df_probe,
                         x="dr_visual_total",
                         y="dr_auditory_total",
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
                     x="dr_visual_total",
                     y="dr_auditory_total",
                     hue="Feature",
                     style='Hemisphere',
                     palette=palette,
                     ax=ax,
                     alpha=0.5)
ax.set_xlim([-0.05, 0.1])
ax.set_ylim([-0.05, 0.1])
ax.legend(bbox_to_anchor=(1.02, 0, 0.3, 1), loc='center left')


