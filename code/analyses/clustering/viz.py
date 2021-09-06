#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 23:17:42 2021

@author: yl254115
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import KernelPCA
from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import TSNE, MDS


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


def plot_DSM(DSM, labels, clustering):
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
    labels = np.asarray(labels)[index].tolist()
    ax = fig.add_axes([0.35, 0.1, 0.6, 0.8])
    clim = [1-S.max(), S.max()]
    clim = [S.mean()-S.std(), 1]
    im = ax.matshow(S, cmap='RdBu_r', clim=clim, origin='lower')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha='left')
    ax.tick_params(labelsize=14)
    axcolor = fig.add_axes([0.96, 0.1, 0.02, 0.8])
    plt.colorbar(im, cax=axcolor)

    return fig, dendro


def plot_dim_reduction(DSM, labels, dendro, colors=None, method='kPCA'):

    S = 1-DSM
    index = dendro['leaves']
    S = S[:, index]  # reorder matrix based on hierarichal clustering
    S = S[index, :]  # reorder matrix based on hierarichal clustering
    labels = np.asarray(labels)[index].tolist()
    if not colors:
        colors = dendro['color_list']
    else:
        colors = np.asarray(colors)[index].tolist()
    # Manifold learning
    if method == 'kPCA':
        kernel_pca = KernelPCA(n_components=2, kernel='precomputed',
                               random_state=0)
        summary = kernel_pca.fit_transform(S)
    elif method == 'tSNE':
        tsne = TSNE(n_components=2, metric='precomputed', random_state=0)
        summary = tsne.fit_transform(DSM)

    # PLOT
    fig_2d, ax = plt.subplots(1, figsize=(40, 30))
    for sel, (color, label) in enumerate(zip(colors, labels)):
        ax.text(summary[sel, 0], summary[sel, 1],
                label, color=color, fontsize=50)
    ax.axis('off')
    ax.set_xlim([np.min(summary[:, 0]), np.max(summary[:, 0])])
    ax.set_ylim([np.min(summary[:, 1]), np.max(summary[:, 1])])

    # Manifold learning
    if method == 'kPCA':
        kernel_pca = KernelPCA(n_components=3, kernel='precomputed',
                               random_state=0)
        summary = kernel_pca.fit_transform(S)
    elif method == 'tSNE':
        tsne = TSNE(n_components=3, metric='precomputed', random_state=0)
        summary = tsne.fit_transform(DSM)

    # PLOT
    fig_3d, ax = plt.subplots(1, figsize=(40, 30))
    #ax = plt.axes(projection='3d')

    #for sel, (color, label) in enumerate(zip(colors, labels)):
    #    ax.text(summary[sel, 0], summary[sel, 1], summary[sel, 2],
    #            label, color=color, fontsize=50)
    ## ax.axis('off')
    #ax.set_xlim([np.min(summary[:, 0]), np.max(summary[:, 0])])
    #ax.set_ylim([np.min(summary[:, 1]), np.max(summary[:, 1])])
    #ax.set_zlim([np.min(summary[:, 2]), np.max(summary[:, 2])])

    return fig_2d, fig_3d
