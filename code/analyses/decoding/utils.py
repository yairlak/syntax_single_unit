#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:56:07 2021

@author: yl254115
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold


def get_3by3_train_test_data(epochs_list, phone_strings, n_splits):
    cv = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
    data_phones = {}
    for ph in phone_strings:
        data_phones[ph] = {}
        queries = [f'(block in [2, 4, 6]) and (phone_string=="{ph}")']
        X, y, stimuli = prepare_data_for_classifier(epochs_list, queries)
        for i_split, (IXs_train, IXs_test) in enumerate(cv.split(X, y)):
            data_phones[ph][i_split] = {}
            data_phones[ph][i_split]['train'] = {}
            data_phones[ph][i_split]['train']['X'] = X[IXs_train]
            data_phones[ph][i_split]['train']['y'] = y[IXs_train]
            data_phones[ph][i_split]['train']['stimuli'] = stimuli[IXs_train]
            data_phones[ph][i_split]['test'] = {}
            data_phones[ph][i_split]['test']['X'] = X[IXs_test]
            data_phones[ph][i_split]['test']['y'] = y[IXs_test]
            data_phones[ph][i_split]['test']['stimuli'] = stimuli[IXs_test]
    return data_phones


def lump_data_together(data, target_ph, vs_phs, i_split, train_test):
    '''
    Cat data together. Set the labels of the target phone to zero,
    And the others' to one.
    Parameters
    ----------
    data : dict
        DESCRIPTION.
    target_ph : str
        Target phone
    vs_phs : list of strings
        list of phones in the other class
    i_split : int
        DESCRIPTION
    train_test: string
        Either 'train' or 'test'

    Returns
    -------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    stimuli : TYPE
        DESCRIPTION.

    '''
    X, y, stimuli = [], [], []
    X.append(data[target_ph][i_split][train_test]['X'])
    y.append(np.zeros_like(data[target_ph][i_split][train_test]['y']))  # 0
    stimuli.extend(data[target_ph][i_split][train_test]['stimuli'])
    for ph in vs_phs:
        X.append(data[ph][i_split][train_test]['X'])
        y.append(np.ones_like(data[ph][i_split][train_test]['y']))  # 1
        stimuli.extend(data[ph][i_split][train_test]['stimuli'])
    X = np.vstack(X)
    y = np.concatenate(y, axis=0)

    return X, y, stimuli


def prepare_data_for_classifier(epochs_list, queries,
                                list_class_numbers=None,
                                min_trials=0):
    '''
    cat epochs data across channel dimension and then prepare for classifier
    '''
    X_all_queries, y_all_queries, stimuli = [], [], []
    for q, query in enumerate(queries):
        if 'heard' in query:  # HACK! for word_string
            continue
        if 'END_OF_WAV' in query:  # HACK! for phone_string
            continue
        if epochs_list[0][query].get_data().shape[0] < min_trials:
            print(f'Less than {min_trials} trials matched query: {query}')
            continue

        X = []
        for epochs in epochs_list:
            if q == 0:
                stimuli.extend(epochs[query].metadata['phone_string'])
            curr_data = epochs[query].get_data()
            X.append(curr_data)
        X = np.concatenate(X, axis=1)  # cat along channel (feature) dimension
        X_all_queries.append(X)
        num_trials = curr_data.shape[0]
        if list_class_numbers:
            class_number = list_class_numbers[q]
        else:
            class_number = q + 1
        y_all_queries.append(np.full(num_trials, class_number))

    # cat along the trial dimension
    X_all_queries = np.concatenate(X_all_queries, axis=0)
    y_all_queries = np.concatenate(y_all_queries, axis=0)

    return X_all_queries, y_all_queries, np.asarray(stimuli)
