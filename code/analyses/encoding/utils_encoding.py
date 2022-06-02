#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:14:16 2022

@author: yl254115
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.stats import fdr_correction
from statsmodels.stats.weightstats import ttest_ind
from mne.stats import spatio_temporal_cluster_1samp_test



def get_mask_fdr(df, block, method,
                 word_or_sentence, r_or_dr, 
                 alpha=0.05):
    
    # No sentence-level stats for evoked
    if word_or_sentence == 'sentence' and method == 'evoked':
        raise()
        
    if word_or_sentence == 'sentence' and r_or_dr == 'r':
        reject_fdr, ps_fdr, scores = get_mask_fdr_sentence_r(df, block,
                                                             alpha)
    elif word_or_sentence == 'sentence' and r_or_dr == 'dr':
        reject_fdr, ps_fdr, scores = get_mask_fdr_sentence_dr(df, block,
                                                              alpha)
    elif word_or_sentence == 'word' and r_or_dr == 'r':
        reject_fdr, ps_fdr, scores = get_mask_fdr_word_r(df, block, method,
                                                         alpha)
    elif word_or_sentence == 'word' and r_or_dr == 'dr':
        reject_fdr, ps_fdr, scores = get_mask_fdr_word_dr(df, block, method,
                                                          alpha)
            
    return reject_fdr, ps_fdr, scores


def get_mask_fdr_sentence_r(df, block, alpha=0.05):
    
    ps = df[f'ps_full_{block}_sentence_trf']
    # FDR ACROSS CHANNELS
    reject_fdr, ps_fdr = fdr_correction(ps,
                                        alpha=alpha,
                                        method='indep')
    # RETURN ALSO CORRESPONDING r's
    rs = df[f'rs_full_{block}_sentence_trf']
    return reject_fdr, ps_fdr, rs


def get_mask_fdr_sentence_dr(df, block, alpha=0.05):
    # GET SCORES
    rs_sentence_full_all_channels = df[f'rs_full_{block}_sentence_per_split_trf'].values
    rs_sentence_feature_all_channels = df[f'rs_feature_{block}_sentence_per_split_trf'].values
    
    # COMPUTE DELTA r's (WITHOUT SPLITS)
    drs_sentence = df[f'rs_full_{block}_sentence_trf'].values - \
                   df[f'rs_feature_{block}_sentence_trf'].values
    
    ps = []
    for rs_sentence_full, rs_sentence_feature in \
        zip(rs_sentence_full_all_channels, rs_sentence_feature_all_channels):
        if rs_sentence_full and rs_sentence_feature:
            # T-TEST ACROSS SPLIT, FOR EACH CHANNEL
            tstat, p, df = ttest_ind(rs_sentence_full, rs_sentence_feature)
            ps.append(p)
        else:
            ps.append(None)
    # FDR ACROSS CHANNELS
    reject_fdr, ps_fdr = fdr_correction(ps, alpha=alpha, method='indep')
    
    return reject_fdr, ps_fdr, drs_sentence


def get_mask_fdr_word_r(df, block, method, alpha=0.05):
    # GET SCORES
    rs_all_channels = df[f'rs_full_{block}_word_{method}']
    rs_splits_all_channels = df[f'rs_full_{block}_word_per_split_{method}']
    
    # FOR EACH CHANNEL, RUN A CLUSTER-BASED PERUMTATION TEST
    clusters_all_channels, ps_clusters_all_channels = [], []
    for rs, rs_splits in zip(rs_all_channels, rs_splits_all_channels):
        rs_splits = np.asarray(rs_splits)
        rs_splits = rs_splits[:, :, None] if rs_splits.ndim == 2 else rs
        
        # CLUSTER-BASED PERMUTATIoN
        _, clusters, ps_clusters, _ = spatio_temporal_cluster_1samp_test(rs_splits,
                                                                out_type='mask',
                                                                n_permutations=1000,
                                                                n_jobs=-1,
                                                                verbose=False,
                                                                tail=1, seed=42)
        clusters_all_channels.append(clusters)
        ps_clusters_all_channels.append(ps_clusters)
        
    # FDR OF CLUSTER SIGNIFCANCE ACROSS CHANNELS
    ps_flatten = [p for l in ps_clusters_all_channels for p in l]
    rejects_clusters_fdr_all_channels_flatten, ps_clusters_fdr_all_channels_flatten = \
        fdr_correction(ps_flatten, alpha=alpha, method='indep')
    
    cnt = 0
    reject_clusters_fdr_all_channels, ps_clusters_fdr_all_channels = [], []
    for ps_clusters in ps_clusters_all_channels: # loop over channels
        if len(ps_clusters)>0:
            rejects_clusters_fdr, ps_clusters_fdr = [], []
            for _ in ps_clusters: # loop over clusters
                reject_cluster_fdr = rejects_clusters_fdr_all_channels_flatten[cnt]
                rejects_clusters_fdr.append(reject_cluster_fdr)
                p_cluster_fdr = ps_clusters_fdr_all_channels_flatten[cnt]
                ps_clusters_fdr.append(p_cluster_fdr)
                cnt += 1
            # APPEND CHANNEL LISTS
            reject_clusters_fdr_all_channels.append(rejects_clusters_fdr)
            ps_clusters_fdr_all_channels.append(ps_clusters_fdr)
        else:
            # APPEND EMPTY
            reject_clusters_fdr_all_channels.append([])
            ps_clusters_fdr_all_channels.append([])
        
    
    # AFTER FDR CORRECTION ACROSS CHANNELS, FOR EACH CHANNEL,
    # COMPUTE MEAN ACROSS TIME (BASED ON FDR-CORRECTED CLUSTERS) OF SIGNIFICANT r's
    rs_significant_mean, reject_fdr = [], []
    for clusters, rejects_clusters_fdr, rs_word in \
        zip(clusters_all_channels, reject_clusters_fdr_all_channels, rs_all_channels): # loop across channels
        rs_significant = []
        for cluster, reject_cluster_fdr \
            in zip(clusters, rejects_clusters_fdr): # loop across clusters
            if reject_cluster_fdr:
                rs_significant.extend(np.asarray(rs_word)[cluster[:, 0]])
        if rs_significant:
            r_significant_mean = np.asarray(rs_significant).mean()
            reject_fdr.append(True)
        else:
            r_significant_mean = 0
            reject_fdr.append(False)
        # APPEND ACROSS CHANNELS
        rs_significant_mean.append(r_significant_mean)
        
    return reject_fdr, ps_clusters_fdr_all_channels, rs_significant_mean


def get_mask_fdr_word_dr(df, block, method, alpha=0.05):
    rs_word_full = df[f'rs_full_{block}_word_per_split_{method}']
    rs_word_feature = df[f'rs_feature_{block}_word_per_split_{method}']
    if rs_word_full and rs_word_feature:
        dr = np.asarray([np.asarray(curr_split_rs_word_full) - np.asarray(curr_split_rs_word_feature) \
                         for (curr_split_rs_word_full, curr_split_rs_word_feature) in \
                         zip(rs_word_full, rs_word_feature)])
        
        # stats function report p_value for each cluster
        dr = dr[:, :, None] if dr.ndim == 2 else dr
        _, clusters, ps, _ = spatio_temporal_cluster_1samp_test(dr, out_type='mask',
                                                                n_permutations=1000, n_jobs=-1,
                                                                verbose=False, tail=1, seed=42)
    
        # FDR
        reject_fdr, ps_fdr = fdr_correction(ps,
                                            alpha=alpha,
                                            method='indep')
        # p_values_ = np.ones_like(dr[0]).T
        # for cluster, pval in zip(clusters, p_values):
        #     p_values_[cluster.T] = pval
    
        # return np.squeeze(p_values_).T
    else:
        return None
    
    return reject_fdr, ps_fdr





def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"



# def get_mask_fdr(df, block, method,
#                  word_or_sentence, r_or_dr, 
#                  alpha=0.05):
#     if word_or_sentence == 'sentence':
#         ps_dr_sentence_trf = df[f'ps_dr_sentence_{block}_trf']
#         reject_fdr, _ = fdr_correction(ps_dr_sentence_trf,
#                                           alpha=alpha,
#                                           method='indep')
#     elif word_or_sentence == 'word':
#         method = 'trf' # !
#         df[f'ps_dr_word_{block}_{method}'] = df.apply(lambda row: compute_stats_for_r_word(row, block, method),
#                                                                 axis=1)
#         ps_dr_word = np.stack(df[f'ps_dr_word_{block}_{method}'].values)
#         reject_fdr_ps_dr_word, _ = fdr_correction(ps_dr_word,
#                                                   alpha=alpha,
#                                                   method='indep')
        
#         reject_fdr_dr = np.asarray([l.any() for l in reject_fdr_ps_dr_word])
#     return reject_fdr_dr


# def compute_stats_for_r_sentence(row, block):
#     rs_sentence_full = row[f'rs_full_{block}_sentence_per_split_trf']
#     rs_sentence_feature = row[f'rs_feature_{block}_sentence_per_split_trf']
#     if rs_sentence_full and rs_sentence_feature:
#         tstat, pval, df = ttest_ind(rs_sentence_full, rs_sentence_feature)
#         return pval
#     else:
#         return None
    
    
# def compute_stats_for_r_word(row, block, method):    
#     """Statistical test applied across CV splits"""
    
#     rs_word_full = row[f'rs_full_{block}_word_per_split_{method}']
#     rs_word_feature = row[f'rs_feature_{block}_word_per_split_{method}']
#     if rs_word_full and rs_word_feature:
#         dr = np.asarray([np.asarray(curr_split_rs_word_full) - np.asarray(curr_split_rs_word_feature) \
#                          for (curr_split_rs_word_full, curr_split_rs_word_feature) in \
#                          zip(rs_word_full, rs_word_feature)])
        
#         # stats function report p_value for each cluster
#         dr = dr[:, :, None] if dr.ndim == 2 else dr
#         _, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(dr, out_type='mask',
#                                                                       n_permutations=1000, n_jobs=-1,
#                                                                       verbose=False, tail=1, seed=42)
    
    
#         p_values_ = np.ones_like(dr[0]).T
#         for cluster, pval in zip(clusters, p_values):
#             p_values_[cluster.T] = pval
    
#         return np.squeeze(p_values_).T
#     else:
#         return None