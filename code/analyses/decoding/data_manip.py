import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
sys.path.append('..')


def get_data(args):
    from utils.data_manip import DataHandler
    data = DataHandler(args.patient, args.data_type, args.filter,
                       args.probe_name, args.channel_name, args.channel_num)
    # Both neural and feature data into a single raw object
    data.load_raw_data(smooth=args.smooth,
                       decimate=args.decimate,
                       verbose=True)
    data.epoch_data(level=args.level,
                    scale_epochs=False,
                    tmin=args.tmin,
                    tmax=args.tmax,
                    verbose=True)
    return data


def add_time_bins(X, k_bins):
    X_new = []
    n_epochs, n_ch, n_times = X.shape
    for t in range(n_times-k_bins+1):
        current_slice = X[:,:,t:(t+k_bins)]
        flat_slice = current_slice.reshape(n_epochs, -1)
        X_new.append(flat_slice)
    X_new = np.dstack(X_new)
    return X_new


def prepare_data_for_classification(epochs_list, queries,
                                    classifier, min_trials=0,
                                    k_bins=1,
                                    equalize_classes=False,
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
                                                             'word_string']].values
        num_trials = len(stimuli_curr_query)
        if num_trials < min_trials:
            print(f'Only {num_trials} trials matched query (less than {min_trials}): {query}')
            continue
        
        
        # GATHER DATA FROM ALL CHANNELS IN EPOCHS_LIST
        X_curr_query_all_epochs = []
        for epochs in epochs_list:
            if epochs is not None:
                X_curr_query_all_epochs.append(epochs[query].get_data())
        X_curr_query_all_epochs = np.concatenate(X_curr_query_all_epochs,
                                                 axis=1) # cat along channel dim
        

        if k_bins > 1:
            X_curr_query_all_epochs = add_time_bins(X_curr_query_all_epochs, k_bins)
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

    # UPSAMPLE CLASSES IF NEEDED
    if equalize_classes == 'upsample':
        largest_class = np.max([X_curr_query.shape[0] for X_curr_query in X])
        X_equalized, y_equalized = [], []
        for X_curr_query, y_curr_query in zip(X, y):
            n_samples = X_curr_query.shape[0]
            if n_samples < largest_class:
                X_curr_query = resample(X_curr_query,
                                        replace=True,
                                        n_samples=largest_class,
                                        random_state=1)
                y_curr_query = resample(y_curr_query,
                                        replace=True,
                                        n_samples=largest_class,
                                        random_state=1)
            X_equalized.append(X_curr_query)
            y_equalized.append(y_curr_query)
        del X, y
    elif equalize_classes == 'downsample':
        smallest_class = np.min([X_curr_query.shape[0] for X_curr_query in X])
        X_equalized, y_equalized = [], []
        for q, (query, stimuli_curr_query, X_curr_query, y_curr_query) in enumerate(zip(queries, stimuli, 
                                                                    X, y)):
            n_samples = X_curr_query.shape[0]
            if n_samples > smallest_class:
                IXs = list(range(n_samples))
                IXs_picked = np.random.choice(IXs, size=smallest_class, replace=False)
                print(IXs_picked)
                X_curr_query = X_curr_query[IXs_picked]
                y_curr_query = y_curr_query[IXs_picked]
            X_equalized.append(X_curr_query)
            y_equalized.append(y_curr_query)
            if verbose:
                print(f'Class {q}, {num_trials} Trials:')
                print(query)
                print(f'Shape of X: {X_curr_query.shape}')
                print(stimuli_curr_query)
        del X, y
    else:
        X_equalized = X
        y_equalized = y
    

    # CAT ALONG TRIAL DIMENSION
    X_equalized = np.concatenate(X_equalized, axis=0)
    y_equalized = np.concatenate(y_equalized, axis=0)
    return X_equalized, y_equalized, stimuli


def get_3by3_train_test_data(epochs_list, phone_strings, n_splits, args):
    cv = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
    data_phones = {}
    for ph in phone_strings:
        data_phones[ph] = {}
        queries = [f'(block in [2, 4, 6]) and (phone_string=="{ph}")']
        X, y, stimuli = prepare_data_for_classification(epochs_list, queries,
                                                        args.classifier, args.min_trials,
                                                        args.equalize_classes,
                                                        verbose=False)
        for i_split, (IXs_train, IXs_test) in enumerate(cv.split(X, y)):
            data_phones[ph][i_split] = {}
            data_phones[ph][i_split]['train'] = {}
            data_phones[ph][i_split]['train']['X'] = X[IXs_train]
            data_phones[ph][i_split]['train']['y'] = y[IXs_train]
            data_phones[ph][i_split]['train']['stimuli'] = stimuli[0, IXs_train, :]
            data_phones[ph][i_split]['test'] = {}
            data_phones[ph][i_split]['test']['X'] = X[IXs_test]
            data_phones[ph][i_split]['test']['y'] = y[IXs_test]
            data_phones[ph][i_split]['test']['stimuli'] = stimuli[0, IXs_test, :]
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

