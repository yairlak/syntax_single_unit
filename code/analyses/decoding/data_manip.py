import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
sys.path.append('..')


def get_data(args):
    from utils.data_manip import DataHandler
    data = DataHandler(args.patient, args.data_type, args.filter,
                   args.probe_name, args.channel_name, args.channel_num)
    # Both neural and feature data into a single raw object
    data.load_raw_data(decimate=args.decimate)
    data.epoch_data(level=args.level,
                    scale_epochs=False,
                    tmin=args.tmin,
                    tmax=args.tmax,
                    smooth=args.smooth,
                    verbose=True)
    return data


def prepare_data_for_classification(epochs_list, queries,
                                    classifier, min_trials=0, 
                                    verbose=False):
    '''
    '''
    # GET SENTENCE-LEVEL DATA BEFORE SPLIT
    X, y, stimuli = [], [], []
    for q, query in enumerate(queries):
        # FILTER QUERIES
        if 'heared' in query: # remove typo for word_string comparison
            continue
        if 'END_OF_WAV' in query: # for phone_string comparison
            continue

        # GET STIMULI FROM FIRST EPOCHS IN LIST
        stimuli_curr_query = epochs_list[0][query].metadata[['sentence_string',
                                                             'word_string']]
        num_trials = len(stimuli_curr_query)
        if num_trials < min_trials:
            print(f'Only {num_trials} trials matched query (less than {min_trials}): {query}')
            continue
        
        
        # GATHER DATA FROM ALL CHANNELS IN EPOCHS_LIST
        X_curr_query_all_epochs = [epochs[query].get_data() for epochs in epochs_list]
        X_curr_query_all_epochs = np.concatenate(X_curr_query_all_epochs,
                                                 axis=1) # cat along channel dim
        if verbose:
            print(f'Class {q}, {num_trials} Trials:')
            print(query)
            print(f'Shape of X: {X_curr_query_all_epochs.shape}')
            print(stimuli_curr_query)
        
        # APPEND ACROSS QUERIES 
        X.append(X_curr_query_all_epochs) 
        
        # 
        if classifier in ['ridge']:
            # Get class value from query (e.g., 'word_length == 4' -> val=4)
            val = float(query.split('==')[1].split()[0])
        else:
            val = q
        y.append(np.full(num_trials, val))
        stimuli.append(stimuli_curr_query)

    # CAT ALONG TRIAL DIMENSION
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y, stimuli


def get_3by3_train_test_data(epochs_list, phone_strings, n_splits):
    cv = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
    data_phones = {}
    for ph in phone_strings:
        data_phones[ph] = {}
        queries = [f'(block in [2, 4, 6]) and (phone_string=="{ph}")']
        X, y, stimuli = prepare_data_for_classification(epochs_list, queries)
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

