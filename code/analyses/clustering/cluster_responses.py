#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 14:07:38 2021

@author: yl254115
"""
from scipy.spatial.distance import pdist, squareform
import argparse
import os
import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import KernelPCA
from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import TSNE, MDS
import sys
import matplotlib.pyplot as plt
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
from utils.utils import dict2filename
from utils.data_manip import load_neural_data, get_events


parser = argparse.ArgumentParser(description='Train a TRF model')
# DATA
parser.add_argument('--patient', action='append', default=['502'])
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike'],
                    action='append', default=['micro'], help='electrode type')
parser.add_argument('--filter', choices=['raw', 'gaussian-kernel',
                                         'gaussian-kernel-25', 'high-gamma'],
                    action='append', default=['gaussian-kernel-10'], help='')
parser.add_argument('--level', choices=['phone', 'word',
                    'sentence-onset', 'sentence-offset'],
                    default='word')
parser.add_argument('--probe-name', default=[], nargs='*',
                    action='append', type=str,
                    help='Probe name to plot (ignores args.channel-name/num)')
parser.add_argument('--channel-name', default=[], nargs='*', action='append',
                    type=str, help='Pick specific channels names')
parser.add_argument('--channe-num', default=[], nargs='*', action='append',
                    type=int, help='If empty list then all channels are taken')
parser.add_argument('--sfreq', default=1000,
                    help='Sampling frequency for both neural and feature data \
                    (must be identical).')
# QUERY
parser.add_argument('--query', default="block in [1,3,5]",
                    help='E.g., limits to first phone in auditory blocks\
                        "and first_phone == 1"')
parser.add_argument('--scale-epochs', default=True, action='store_true',
                    help='If true, data is scaled *after* epoching')
# MODEL
parser.add_argument('--metric', default='correlation',
                    choices=['spike_count', 'correlation'])
# MISC
parser.add_argument('--tmin', default=-0.6, type=float,
                    help='Start time of word time window')
parser.add_argument('--tmax', default=0.8, type=float,
                    help='End time of word time window')
parser.add_argument('--tmin_window', default=-0.1, type=float,
                    help='Start time of receptive-field kernel')
parser.add_argument('--tmax_window', default=0.5, type=float,
                    help='End time of receptive-field kernel')
# PATHS
parser.add_argument('--path2figures',
                    default=os.path.join
                    ('..', '..', '..', 'Figures', 'clustering'),
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
# FNAME
list_args2fname = ['patient', 'data_type', 'filter', 'model_type',
                   'probe_name', 'ablation_method', 'query']
args2fname = args.__dict__.copy()
fname = dict2filename(args2fname, '_', list_args2fname, '', True)
print(fname)
if not os.path.exists(args.path2figures):
    os.makedirs(args.path2figures)

####################
# LOAD NEURAL DATA #
####################
epochs_neural_sentence = load_neural_data(args)
assert len(epochs_neural_sentence) == 1
epochs_neural_sentence = epochs_neural_sentence[0]

epochs_neural_sentence.crop(args.tmin, args.tmax)

words = sorted(list(set(epochs_neural_sentence.metadata['word_string'])))
X = []
for word in words:
    x_i = epochs_neural_sentence[f'word_string=="{word}"'].\
                        get_data().mean(axis=0)
    X.append(x_i)
X = np.asarray(X)
n_words, n_channels, n_times = X.shape
for ch, ch_name in enumerate(epochs_neural_sentence.ch_names):
    DSM = squareform(pdist(X[:, ch, :], 'correlation'))

    ############################
    # CLUSTER CONFUSION MATRIX #
    ############################
    linkage = 'complete'
    # Setting distance_thershold = 0 ensures we compute the full tree
    clustering = AgglomerativeClustering(linkage=linkage,
                                         n_clusters=None,
                                         distance_threshold=0,
                                         affinity='precomputed')
    clustering.fit(DSM)

    #####################
    # PLOT MAT + DENDRO #
    #####################

    def plot_dendrogram(model, **kwargs):
        #  Create linkage matrix and then plot the dendrogram
        #  create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)
        # Plot the corresponding dendrogram
        dendro = dendrogram(linkage_matrix, **kwargs)
        return dendro

    # PLOT DENDRO
    fig = plt.figure(figsize=(40, 30))
    ax_dendro = fig.add_axes([0.09, 0.1, 0.2, 0.8])
    dendro = plot_dendrogram(clustering, ax=ax_dendro, orientation='left')
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])

    # PLOT SIMILARITY MATRIX
    index = dendro['leaves']
    S = 1-DSM
    S = S[:, index]  # reorder matrix based on hierarichal clustering
    S = S[index, :]  # reorder matrix based on hierarichal clustering
    labels = np.asarray(words)[index].tolist()
    ax = fig.add_axes([0.35, 0.1, 0.6, 0.8])
    clim = [1-S.max(), S.max()]
    im = ax.matshow(S, cmap='RdBu_r', clim=clim, origin='lower')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha='left')
    ax.tick_params(labelsize=14)
    axcolor = fig.add_axes([0.96, 0.1, 0.02, 0.8])
    plt.colorbar(im, cax=axcolor)

    args2fname = args.__dict__.copy()
    if len(list(set(args2fname['data_type']))) == 1:
        args2fname['data_type'] = list(set(args2fname['data_type']))
    if len(list(set(args2fname['filter']))) == 1:
        args2fname['filter'] = list(set(args2fname['filter']))
    args2fname['probe_name'] = ch_name
    fname_fig = dict2filename(args2fname, '_', list_args2fname, 'png', True)
    fname_fig_rsa = os.path.join(args.path2figures, f'DSM_{fname_fig}')
    fig.savefig(fname_fig_rsa)
    print('Figures saved to: ' + fname_fig_rsa)
    plt.close('all')

    #########
    # t-SNE #
    #########
    kernel_pca = KernelPCA(n_components=2, kernel='precomputed',
                           random_state=0)
    summary = kernel_pca.fit_transform(S)
    # PLOT
    fig, ax = plt.subplots(1, figsize=(40, 30))
    colors = dendro['color_list']
    for sel, (color, label) in enumerate(zip(colors, labels)):
        ax.text(summary[sel, 0], summary[sel, 1],
                label, color=color, fontsize=50)
    ax.axis('off')
    ax.set_xlim([np.min(summary[:, 0]), np.max(summary[:, 0])])
    ax.set_ylim([np.min(summary[:, 1]), np.max(summary[:, 1])])
    # plt.tight_layout()
    fname_fig_tsne = os.path.join(args.path2figures,
                                  f'kernelPCA_2D_{fname_fig}')
    fig.savefig(fname_fig_tsne)

    kernel_pca = KernelPCA(n_components=3, kernel='precomputed',
                           random_state=0)
    summary = kernel_pca.fit_transform(S)
    # PLOT
    fig, ax = plt.subplots(1, figsize=(40, 30))
    ax = plt.axes(projection='3d')
    colors = dendro['color_list']
    for sel, (color, label) in enumerate(zip(colors, labels)):
        ax.text(summary[sel, 0], summary[sel, 1], summary[sel, 2],
                label, color=color, fontsize=50)
    # ax.axis('off')
    ax.set_xlim([np.min(summary[:, 0]), np.max(summary[:, 0])])
    ax.set_ylim([np.min(summary[:, 1]), np.max(summary[:, 1])])
    ax.set_zlim([np.min(summary[:, 2]), np.max(summary[:, 2])])
    # plt.tight_layout()
    fname_fig_tsne = os.path.join(args.path2figures,
                                  f'kernelPCA_3D_{fname_fig}')
    fig.savefig(fname_fig_tsne)
