#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 17:29:59 2021

@author: yl254115
"""

import os, sys, copy
import mne
import numpy as np
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
from utils.data_manip import load_neural_data
from mne.decoding import ReceptiveField, TimeDelayingRidge
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from tqdm import tqdm

def train_TRF(X_train, y_train, sfreq_down_sampled, args):
    # DIMS
    n_timepoints, n_epochs, n_outputs = y_train.shape
    
    if args.model_type in ['ridge', 'lasso']:
        alphas = np.logspace(-3, 8, 10)
        if args.model_type == 'ridge':
            estimator = linear_model.RidgeCV(alphas=alphas, alpha_per_target=True)
        elif args.model_type == 'lasso':
            estimator = linear_model.LassoCV()
       
        rf = ReceptiveField(args.tmin_rf, args.tmax_rf, sfreq_down_sampled, estimator=estimator, scoring='r2', n_jobs=-1) #feature_names=feature_values, 
        rf.fit(X_train, y_train)
    
    elif args.model_type == 'ridge_laplacian':
        alphas = np.logspace(-3, 5, 10)
        inner_cv = KFold(n_splits=args.n_folds_inner, shuffle=True, random_state=0)
        scores = np.zeros((args.n_folds_inner, alphas.size, n_outputs))
        models = [] # list (len=num_splits) of lists, each sublist its len is num_alphas with the corresponding TRF models
        for i_split, (train, test) in enumerate(inner_cv.split(X_train.transpose([1, 2, 0]), y_train.transpose([1, 2, 0]))):     
            models_curr_seed = []
            for i_alpha, alpha in enumerate(alphas):
                print(f'Fitting TRF with Laplacian regularization (alpha={alpha})')
                estimator = TimeDelayingRidge(args.tmin_rf, args.tmax_rf, sfreq_down_sampled, reg_type=['laplacian', 'ridge'], alpha=alpha, n_jobs=-1)
                rf = ReceptiveField(args.tmin_rf, args.tmax_rf, sfreq_down_sampled, estimator=estimator, scoring='r2', n_jobs=-1) #feature_names=feature_values,                 
                rf.fit(X_train[:, train, :], y_train[:, train, :])
                
                # Make predictions about the model output, given input stimuli.
                scores[i_split, i_alpha, :] = rf.score(X_train[:, test, :], y_train[:, test, :])
                models_curr_seed.append(rf)
            models.append(models_curr_seed)
        
        ix_best_alpha_lap = np.argmax(scores.mean(axis=0), axis=0) # average across seeds and pick maximal alpha per electrode (shape = n_outputs)
        best_alpha = alphas[ix_best_alpha_lap]
        print(f'Best alpha={best_alpha}')
     
    return rf


def get_test_data_word(raw_features, sentences_test, blocks_test, args):
    # LOAD NEURAL DATA
    args_temp = copy.deepcopy(args)
    args_temp.decimate = []
    args_temp.level = 'word'
    epochs_neural_word = load_neural_data(args_temp)[0]
    
    # EPOCH FEATURE DATA
    epochs_features_word = mne.Epochs(raw_features, epochs_neural_word.events, event_id=epochs_neural_word.event_id, tmin=epochs_neural_word.tmin, tmax=epochs_neural_word.tmax, metadata=epochs_neural_word.metadata, baseline=None)
    epochs_features_word = epochs_features_word[args.query]
    
    # TAKE ONLY TEST WORDS
    query2test_sentences = ' or '.join([f'(sentence_string=="{s}" and block=={b})' for s, b in zip(sentences_test, blocks_test)])
    epochs_features_word = epochs_features_word[query2test_sentences]
    epochs_neural_word = epochs_neural_word[query2test_sentences]
    
    # DECIMATE
    if args.decimate:
        epochs_neural_word = epochs_neural_word.decimate(args.decimate)
        epochs_features_word = epochs_features_word.decimate(args.decimate)

    # EXTRACT DATA
    X_word = epochs_features_word.get_data().transpose([2,0,1])
    y_word = epochs_neural_word.get_data().transpose([2,0,1])
    
    return X_word, y_word, epochs_neural_word.times


def eval_TRF_across_epochs(rf, X_test, y_test, valid_samples, args):
      
    y_pred = rf.predict(X_test) # num_timepoints X num_epochs X num_electrodes
    y_pred = y_pred[valid_samples]
    y_masked = y_test[valid_samples]
    
    n_times, n_epochs, n_outputs = y_masked.shape
    scores = []
    for t in tqdm(range(n_times)):
        curr_score = r2_score(y_masked[t,:,:], y_pred[t,:,:], multioutput='raw_values')
        scores.append(curr_score)
       
    
    return np.asarray(scores)