#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 17:29:59 2021

@author: yl254115
"""

import numpy as np
from encoding.models import TimeDelayingRidgeCV
from mne.decoding import ReceptiveField
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import linear_model
from sklearn.metrics import r2_score
#from tqdm import tqdm


def get_scalers(X, method='standard'):
    # ASSUMING LAST DIM IS FEATURE OR OUTPUT (CHANNEL)
    # X: n_times X n_trials X n_features (or last dim is n_outputs)

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()

    scalers = []
    for i_feat in range(X.shape[-1]):
        vec = X[:, :, i_feat].reshape(-1, 1)
        scaler.fit(vec)
        scalers.append(scaler)

    return scalers


def scale_data(X, feature_names, scalers,
               features_without_scaling=['is_first_word', 'is_first_phone']):
    # ASSUMING LAST DIM IS FEATURE OR OUTPUT (CHANNEL)
    # X: n_times X n_trials X n_features (or last dim is n_outputs)

    n_times, n_trials, n_features = X.shape
    for i_feat in range(n_features):
        if feature_names:  # for neural data, instead of a list should be None
            feature_name = feature_names[i_feat]
            if feature_name in features_without_scaling:
                continue  # skip current feature if its without scaling
        scaler = scalers[i_feat]
        vec = X[:, :, i_feat].reshape(-1, 1)
        X[:, :, i_feat] = scaler.transform(vec).reshape(n_times, n_trials)
    return X


def train_TRF(X_train, y_train, sfreq, args):
    # DIMS
    n_timepoints, n_epochs, n_outputs = y_train.shape
    if args.model_type == 'ridge':
        alphas = np.logspace(-3, 8, 100)
        estimator = linear_model.RidgeCV(alphas=alphas, alpha_per_target=True)
    elif args.model_type == 'lasso':
        alphas = np.logspace(-3, 8, 100)
        estimator = linear_model.LassoCV()
    elif args.model_type == 'ridge_laplacian':
        alphas = np.logspace(-3, 5, 3)
        estimator = TimeDelayingRidgeCV(tmin=args.tmin_rf,
                                        tmax=args.tmax_rf,
                                        sfreq=sfreq,
                                        alphas=alphas,
                                        cv=args.n_folds_inner,
                                        reg_type=['laplacian', 'ridge'],
                                        alpha_per_target=True)
    rf = ReceptiveField(args.tmin_rf, args.tmax_rf, sfreq,
                        estimator=estimator, scoring='corrcoef', n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf


def eval_TRF_across_epochs(rf, X_test, y_test, valid_samples, args):
    y_pred = rf.predict(X_test)  # n_times X n_epochs X n_electrodes
    y_pred = y_pred[valid_samples]
    y_masked = y_test[valid_samples]
    n_times, n_epochs, n_outputs = y_masked.shape
    scores = []
    for t in range(n_times):
        curr_score = r2_score(y_masked[t, :, :], y_pred[t, :, :],
                              multioutput='raw_values')
        scores.append(curr_score)
    return np.asarray(scores)


def reduce_design_matrix(X, feature_name, feature_info, ablation_method,
                         start_sample=0):
    '''

    Parameters
    ----------
    X : ndarray num_timepoint X num_epochs X num_features
        Design matrix.
    start_sample : int
        Sample number from which to start the shuffle/zero.
    feature_name : string
        DESCRIPTION.
    feature_info : dict
        DESCRIPTION.
    ablation_method : str
        one of remove/zero/shuffle.

    Returns
    -------
    X_reduced : ndarray num_timepoint X num_epochs X num_features
        Reduced design matrix.

    '''
    if feature_name == 'full':
        X_reduced = X
    else:
        # Find index of target feature info in design matrix
        if feature_name in feature_info.keys():  # ablate all feature values
            st, ed = feature_info[feature_name]['IXs']
        else:  # ablate a specific feature value
            for k in feature_info.keys():
                if feature_name in feature_info[k]['names']:
                    break
            IX = feature_info[k]['names'].index(feature_name)
            st = feature_info[k]['IXs'][0] + IX
            ed = st + 1
        # Three ways to ablate feature info
        if ablation_method == 'remove':
            X_reduced = np.delete(X, range(st, ed), 2)
        elif ablation_method == 'zero':
            X_reduced = X.copy()
            X_reduced[start_sample:, :, st:ed] = 0
        elif ablation_method == 'shuffle':
            X_reduced = X.copy()
            X_reduced_FOI = X_reduced[start_sample:, :, st:ed]
            X_reduced_FOI[X_reduced_FOI != 0] = \
                np.random.permutation(X_reduced_FOI[X_reduced_FOI != 0])
            X_reduced[start_sample:, :, st:ed] = X_reduced_FOI
    return X_reduced
