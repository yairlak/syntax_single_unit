#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 21:29:29 2022

@author: yair
"""

from .data_manip import prepare_data_for_classification
from .models import define_model
from sklearn.model_selection import LeaveOneOut, KFold
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def decode_comparison(epochs_list, comparisons, args, n_perm=10000):
    # PREPARE DATA FOR DECODING
    print('\nPERPARING DATA FOR CLASSIFICATION:')
    X, y, stimuli= prepare_data_for_classification(epochs_list,
                                                   comparisons[0]['queries'],
                                                   args.classifier,
                                                   args.min_trials,
                                                   equalize_classes=args.equalize_classes,
                                                   verbose=True)
    stimuli_gen = []
    if args.GAC or args.GAM:
        if args.GAC: print('-'*30, '\nGeneralization Across Conditions\n', '-'*30)
        if args.GAM: print('-'*30, '\nGeneralization Across Modalities\n', '-'*30)
        X_gen, y_gen, stimuli_gen= prepare_data_for_classification(epochs_list,
                                                                   comparisons[1]['queries'],
                                                                   args.classifier,
                                                                   args.min_trials,
                                                                   args.k_bins,
                                                                   equalize_classes=args.equalize_classes,
                                                                   verbose=True)
        print(stimuli_gen)
    classes = sorted(list(set(y)))
       
    # SET A MODEL (CLASSIFIER)
    clf, temp_estimator = define_model(args)
    
    
    # LEAVE-ONE-OUT EVALUATION 
    print('\n', '-'*40, 
          f'\nTraining a {args.classifier} model for a {len(list(set(y)))}-class problem\n', '-'*40)
    loo = KFold(X.shape[0], shuffle=True, random_state=1)
    y_hats, y_trues, = [], []
    for i_split, (IXs_train, IX_test) in enumerate(loo.split(X, y)):
        print(f'Split {i_split+1}/{X.shape[0]}')
        # TRAIN MODEL
        if (args.GAC or args.GAM) and i_split == 0: # Use all training data once
            print(f'Training model')
            temp_estimator.fit(X, y)
        else:
            temp_estimator.fit(X[IXs_train], y[IXs_train])
    
        # PREDICT
        if (args.GAC or args.GAM): # Eval on each test sample (LOO-like)
            print(f'Predict labels')
            proba = temp_estimator.predict_proba(X_gen[IX_test])
            y_hats.append(np.squeeze(proba))
            y_trues.append(np.squeeze(y_gen[IX_test]))
        else:
            proba = temp_estimator.predict_proba(X[IX_test])
            y_hats.append(np.squeeze(proba))
            y_trues.append(y[IX_test])
    y_hats = np.asarray(y_hats)  # n_samples X n_timepoints
    y_trues = np.asarray(y_trues).squeeze()  # n_samples
    
    ##############
    # EVAL MODEL #
    ##############
    print('Eval model')
    # AUC
    if (args.GAC or args.GAM or args.multi_class):
        multi_class = 'ovr'  # one-vs-rest
    else:
        multi_class = 'raise'
    
    scores, pvals = [], []
    
    for i_t in tqdm(range(y_hats.shape[1])):  # loop over n_times
        if args.multi_class:
            scores_true = roc_auc_score(y_trues, y_hats[:, i_t, :],
                                        multi_class=multi_class,
                                        average='weighted')
        else: # Binary case
            scores_true = roc_auc_score(y_trues, y_hats[:, i_t, 1],
                                    multi_class=multi_class,
                                    average='weighted')
        scores_perm = []
        for i_perm in range(n_perm):
            y_perm = y_trues[np.random.permutation(y_trues.size)]
            if args.multi_class:
                scores_perm.append(roc_auc_score(y_perm, y_hats[:, i_t, :],
                                                 multi_class=multi_class,
                                                 average='macro'))
            else:
                scores_perm.append(roc_auc_score(y_perm, y_hats[:, i_t, 1],
                                                 multi_class=multi_class,
                                                 average='macro'))
            
        C = sum(np.asarray(scores_perm) > scores_true)
        pval = (C + 1) / (n_perm + 1)
        scores.append(scores_true)
        pvals.append(pval)
    
    # The shape of scores is: num_splits X num_timepoints ( X num_timepoints)
    scores = np.asarray(scores).squeeze()
    pvals = np.asarray(pvals)
    
    return scores, pvals, temp_estimator, clf, stimuli, stimuli_gen
