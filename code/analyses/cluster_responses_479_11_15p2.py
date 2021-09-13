#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 14:07:38 2021

@author: yl254115
"""
from scipy.spatial.distance import pdist, squareform
import argparse
import os
import pickle
import datetime
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from utils.utils import dict2filename
from utils.data_manip import DataHandler
from clustering.viz import plot_DSM, plot_dim_reduction

parser = argparse.ArgumentParser(description='Train a TRF model')
# DATA
parser.add_argument('--patient', action='append', default=['479_11'])
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike'],
                    action='append', default=['spike'], help='electrode type')
parser.add_argument('--filter', action='append', default=['raw'],
                    help='raw/high-gamma')
parser.add_argument('--smooth', default=25, help='')
parser.add_argument('--level', choices=['phone', 'word',
                    'sentence-onset', 'sentence-offset'],
                    default='word')
parser.add_argument('--probe-name', default=['LSTG'], nargs='*',
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
parser.add_argument('--query', default="block in [2,4,6] and word_string.str.len()>1",
                    help='E.g., limits to first phone in auditory blocks\
                        "and first_phone == 1"')
parser.add_argument('--min-trials', default=0,
                    help='Minimum number of trials.')
#parser.add_argument('--scale-epochs', default=True, action='store_true',
#                    help='If true, data is scaled *after* epoching')
# MODEL
parser.add_argument('--metric', default='correlation',
                    choices=['spike_count', 'correlation'])
# MISC
parser.add_argument('--tmin', default=0.05, type=float,
                    help='Start time of word time window')
parser.add_argument('--tmax', default=0.35, type=float,
                    help='End time of word time window')
# PATHS
parser.add_argument('--path2figures',
                    default=os.path.join
                    ('..', '..', 'Figures', 'clustering'),
                    help="Path to where trained models and results are saved")


begin_time = datetime.datetime.now()
np.random.seed(1)
#############
# USER ARGS #
#############
args = parser.parse_args()
args.patient = ['patient_' + p for p in args.patient]
args.block_type = 'both'
print('args\n', args)
assert len(args.patient) == len(args.data_type) == len(args.filter)
list_args2fname = ['patient', 'data_type', 'filter', 'smooth', 'model_type',
                   'probe_name', 'ablation_method', 'query']
args2fname = args.__dict__.copy()
if len(list(set(args2fname['data_type']))) == 1:
    args2fname['data_type'] = list(set(args2fname['data_type']))
if len(list(set(args2fname['filter']))) == 1:
    args2fname['filter'] = list(set(args2fname['filter']))
if not os.path.exists(args.path2figures):
    os.makedirs(args.path2figures)

#############
# LOAD DATA #
#############
data = DataHandler(args.patient, args.data_type, args.filter,
                   args.probe_name, args.channel_name, args.channel_num,
                   feature_list=None)
# LOAD RAW DATA
data.load_raw_data()
# EPOCH DATA
data.epoch_data(level=args.level,
                query=args.query,
                smooth=args.smooth,
                verbose=True)
assert len(data.epochs) == 1

words = sorted(list(set(data.epochs[0].metadata['word_string'])))
X = []
labels = []
colors = []
for word in words:
    x_i = data.epochs[0][f'word_string=="{word}"'].crop(args.tmin, args.tmax).\
                        get_data()
    if x_i.shape[0] > args.min_trials:
        X.append(np.mean(x_i, axis=0))
        labels.append(word)
        #  if 'y' in word or 'v' in word or 'k' in word or 'w' in word or 'W' in word or 'x' in word:
        if 'w' in word or 'W' in word:
            color = 'g'
        else:
            color = 'k'
        colors.append(color)
X = np.asarray(X)
n_words, n_channels, n_times = X.shape
DSMs = []
linkage = 'complete'
ch_names = data.epochs[0].ch_names
for ch, ch_name in enumerate(ch_names):
    DSM = squareform(pdist(X[:, ch, :], 'correlation'))
    DSM[np.isnan(DSM)] = DSM[~np.isnan(DSM)].max()
    DSMs.append(DSM)
    ############################
    # CLUSTER CONFUSION MATRIX #
    ############################
    # Setting distance_thershold = 0 ensures we compute the full tree
    clustering = AgglomerativeClustering(linkage=linkage,
                                         n_clusters=None,
                                         distance_threshold=0,
                                         affinity='precomputed')
    clustering.fit(DSM)

    #####################
    # PLOT MAT + DENDRO #
    #####################
    fig_DSM, dendro = plot_DSM(DSM, labels, clustering)
    args2fname['probe_name'] = ch_name
    fname_fig = dict2filename(args2fname, '_', list_args2fname, 'png', True)
    fig_DSM.savefig(os.path.join(args.path2figures, f'DSM_{fname_fig}'))
    plt.close(fig_DSM)

    ##############
    # KERNEL-PCA #
    ##############
    fig_2d, fig_3d = plot_dim_reduction(DSM, labels, dendro, colors,
                                        method='kPCA')
    fname_fig_2d = os.path.join(args.path2figures, f'kernelPCA_2D_custom_{fname_fig}')
    fig_2d.savefig(fname_fig_2d)
    plt.close(fname_fig_2d)
    fname_fig_3d = os.path.join(args.path2figures, f'kernelPCA_3D_custom_{fname_fig}')
    fig_3d.savefig(fname_fig_3d)
    plt.close(fname_fig_3d)

    ########
    # tSNE #
    ########
    fig_2d, fig_3d = plot_dim_reduction(DSM, labels, dendro, colors,
                                        method='tSNE')
    fname_fig_2d = os.path.join(args.path2figures, f'tSNE_2D_custom_{fname_fig}')
    fig_2d.savefig(fname_fig_2d)
    plt.close(fname_fig_2d)
    fname_fig_3d = os.path.join(args.path2figures, f'tSNE_3D_custom_{fname_fig}')
    fig_3d.savefig(fname_fig_3d)
    plt.close(fname_fig_3d)

    print('Figures saved to: ' +
          os.path.join(args.path2figures, f'\nDSM_{fname_fig}') +
          os.path.join(args.path2figures, f'\nkernelPCA_2D_{fname_fig}') +
          os.path.join(args.path2figures, f'\nkernelPCA_3D_{fname_fig}'))

    fname = f'{args.patient[0]}_{args.data_type[0]}_{args.filter[0]}_{args.smooth}_{ch_name}_{args.query}.clu'
    fname = f'../../Output/clustering/{fname}'
    with open(fname, 'wb') as f:
        pickle.dump([DSM, clustering, dendro, labels, ch_name, args], f)

# FOR ALL CHANNELS TOGETHER
DSMs = np.asarray(DSMs)
DSMs = np.sqrt((DSMs**2).sum(axis=0))
clustering = AgglomerativeClustering(linkage=linkage,
                                     n_clusters=None,
                                     distance_threshold=0,
                                     affinity='precomputed')
clustering.fit(DSMs)
fig_DSM, dendro = plot_DSM(DSMs, labels, clustering)
args2fname['probe_name'] = args.probe_name
fname_fig = dict2filename(args2fname, '_', list_args2fname, 'png', True)
fig_DSM.savefig(os.path.join(args.path2figures, f'DSM_{fname_fig}'))
plt.close(fig_DSM)

fig_2d, fig_3d = plot_dim_reduction(DSMs, labels, dendro, colors, method='kPCA')
fname_fig_2d = os.path.join(args.path2figures, f'kernelPCA_custom_2D_{fname_fig}')
fig_2d.savefig(fname_fig_2d)
plt.close(fname_fig_2d)
fname_fig_3d = os.path.join(args.path2figures, f'kernelPCA_custom_3D_{fname_fig}')
fig_3d.savefig(fname_fig_3d)
plt.close(fname_fig_3d)

fig_2d, fig_3d = plot_dim_reduction(DSMs, labels, dendro, colors, method='tSNE')
fname_fig_2d = os.path.join(args.path2figures, f'tSNE_2D_custom_{fname_fig}')
fig_2d.savefig(fname_fig_2d)
plt.close(fname_fig_2d)
fname_fig_3d = os.path.join(args.path2figures, f'tSNE_3D_custom_{fname_fig}')
fig_3d.savefig(fname_fig_3d)
plt.close(fname_fig_3d)
