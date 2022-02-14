#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:26:45 2021

@author: yl254115
"""

from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.svm import LinearSVC
from mne.decoding import (cross_val_multiscore, LinearModel, GeneralizingEstimator, SlidingEstimator)
from sklearn.multiclass import OneVsRestClassifier

def define_model(args, verbose=False):
    if args.multi_class:
        if verbose: print('Multiclass classification')
        if args.classifier == 'logistic':
            clf = make_pipeline(OneVsRestClassifier(LogisticRegression(solver='lbfgs',
                                                                       class_weight='balanced',
                                                                       multi_class='ovr')))
        elif args.classifier == 'svc':
            clf = make_pipeline(OneVsRestClassifier(LinearSVC(class_weight='balanced')))
        elif args.classifier == 'ridge':
            clf = make_pipeline(OneVsRestClassifier(Ridge()))
        elif args.classifier == 'ridge_classifier':
            clf = make_pipeline(OneVsRestClassifier(RidgeClassifier(class_weight='balanced')))
        # ESTIMATOR
        if args.gat:
            time_gen = GeneralizingEstimator(clf, scoring='roc_auc_ovo', n_jobs=-1)
        else:
            time_gen = SlidingEstimator(clf, n_jobs=-1, scoring='roc_auc_ovo', verbose=True)
    else:
        if verbose:
            print('Binary classification')
        if args.classifier == 'logistic':
            clf = make_pipeline(LinearModel(LogisticRegression(C=1,
                                                               solver='liblinear',
                                                               class_weight='balanced')))
        elif args.classifier == 'svc':
            clf = make_pipeline(LinearSVC(class_weight='balanced'))
        elif args.classifier == 'ridge':
            clf = make_pipeline(Ridge())
        elif args.classifier == 'ridge_classifier':
            clf = make_pipeline(RidgeClassifier(class_weight='balanced'))
        
        # ESTIMATOR
        if args.gat:
            time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=-1)
        else:
            time_gen = SlidingEstimator(clf, scoring='roc_auc', n_jobs=-1, verbose=True)
    
    return clf, time_gen
