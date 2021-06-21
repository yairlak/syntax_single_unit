import sys, os, pickle, glob
import mne
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg')


def prepare_data_for_GAT(args):
    '''
    :param patients: (list) 
   :param hospitals: (list) same len as patients
    :param picks_all_patients: (list of channel numbers or 'all') same len as patients
    :param query_classes_train: (list) queries for each class
    :param query_classes_test: (optional - list) queries for each class (if empry, then 5-fold CV is used)
    :param root_path:
    :param k: (scalar) number of subsequent time points to cat
    :return:
    1. times
    2. X_train_query
    3. y_train_query
    4. X_test_query
    5. y_test_query
    '''

    #print(args)
    patients=args.patients
    hospitals=args.hospitals
    picks_micro=args.picks_micro # LIST with #patients SUBLISTS with picks per patient.
    picks_macro=args.picks_macro
    picks_spike=args.picks_spike
    
    query_classes_train=args.train_queries; query_classes_test=args.test_queries; 
    root_path=args.root_path
    k=args.cat_k_timepoints

    # Times
    train_times = {}
    train_times["start"] = -1
    train_times["stop"] = 1
    # train_times["step"] = 0.01

    X_train = [[] for _ in query_classes_train]
    y_train = []
    X_test, y_test = ([], [])
    if query_classes_test:
        X_test = [[] for _ in query_classes_test]
        y_test = []
    print(picks_micro)
    for i, (patient, hospital, pick_micro, pick_macro, pick_spike) in enumerate(zip(patients, hospitals, picks_micro, picks_macro, picks_spike)):
        # ----------------------------------------
        # --------- micro high-gamma ------------
        # ---------------------------------------
        epochs_filenames = glob.glob(os.path.join(root_path, 'Data', hospital, patient, 'Epochs', '*_micro_*.h5'))
        epochs_filenames = filter_relevant_epochs_filenames(epochs_filenames, pick_micro)
        for c, fn in enumerate(sorted(epochs_filenames)):
            print('-'*100)
            print('Loading TRAIN epochs object: %s' % (os.path.basename(fn)))
            print('-'*100)
            try:
                epochsTFR = mne.time_frequency.read_tfrs(os.path.join(root_path, 'Data', hospital, patient, 'Epochs', fn))[0]
                print(epochsTFR)
                curr_epochs = mne.EpochsArray(np.average(epochsTFR.data, axis=2), epochsTFR.info, tmin=np.min(epochsTFR.times), metadata=epochsTFR.metadata, events=epochsTFR.events, event_id=epochsTFR.event_id)
                del epochsTFR
                print(curr_epochs)
                print('previous sfreq: %f' % curr_epochs.info['sfreq'])
                curr_epochs = curr_epochs.resample(100, npad='auto')
                print('new sfreq: %f' % curr_epochs.info['sfreq'])
                X_train, X_test, curr_epochs_query = get_train_test_data_from_epochs(curr_epochs, query_classes_train, query_classes_test, X_train, X_test, train_times)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print('!!!!!! ERROR !!!!!!: %s \n %s line %s' % (fn, e, exc_tb.tb_lineno))
        # ---------------------------------------
        # --------- macro high-gamma ------------
        # ---------------------------------------
        epochs_filenames = glob.glob(os.path.join(root_path, 'Data', hospital, patient, 'Epochs', '*_macro_*.h5'))
        epochs_filenames = filter_relevant_epochs_filenames(epochs_filenames, pick_macro)
        for c, fn in enumerate(sorted(epochs_filenames)):
            print('-'*100)
            print('Loading TRAIN epochs object: %s' % (os.path.basename(fn)))
            print('-'*100)
            try:
                epochsTFR = mne.time_frequency.read_tfrs(os.path.join(root_path, 'Data', hospital, patient, 'Epochs', fn))[0]
                print(epochsTFR)
                curr_epochs = mne.EpochsArray(np.average(epochsTFR.data, axis=2), epochsTFR.info, tmin=np.min(epochsTFR.times), metadata=epochsTFR.metadata, events=epochsTFR.events, event_id=epochsTFR.event_id)
                del epochsTFR
                print('previous sfreq: %f' % curr_epochs.info['sfreq'])
                curr_epochs = curr_epochs.resample(100, npad='auto')
                print('new sfreq: %f' % curr_epochs.info['sfreq'])
                X_train, X_test, curr_epochs_query = get_train_test_data_from_epochs(curr_epochs, query_classes_train, query_classes_test, X_train, X_test, train_times)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print('!!!!!! ERROR !!!!!!: %s \n %s line %s' % (fn, e, exc_tb.tb_lineno))

        # ---------------------------------------
        # --------- single-unit -----------------
        # ---------------------------------------
        epochs_filenames = glob.glob(os.path.join(root_path, 'Data', hospital, patient, 'Epochs', '*_spikes_*.h5'))
        epochs_filenames = filter_relevant_epochs_filenames(epochs_filenames, pick_spike)
        #print(epochs_filenames)
        for c, fn in enumerate(sorted(epochs_filenames)):
            print('-'*100)
            print('Loading TRAIN epochs object: %s' % (os.path.basename(fn)))
            print('-'*100)
            try:
                epochsTFR = mne.time_frequency.read_tfrs(os.path.join(root_path, 'Data', hospital, patient, 'Epochs', fn))[0]
                print(epochsTFR)
                curr_epochs = mne.EpochsArray(np.average(epochsTFR.data, axis=2), epochsTFR.info, tmin=np.min(epochsTFR.times), metadata=epochsTFR.metadata, events=epochsTFR.events, event_id=epochsTFR.event_id)
                del epochsTFR
                print('previous sfreq: %f' % curr_epochs.info['sfreq'])
                curr_epochs = curr_epochs.resample(100, npad='auto')
                print('new sfreq: %f' % curr_epochs.info['sfreq'])
                X_train, X_test, curr_epochs_query = get_train_test_data_from_epochs(curr_epochs, query_classes_train, query_classes_test, X_train, X_test, train_times)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print('!!!!!! ERROR !!!!!!: %s \n %s line %s' % (fn, e, exc_tb.tb_lineno))

    print('X_train shape:')
    [print(x.shape) for x in X_train[0]]
    X_train = [np.concatenate(d, axis=1) for d in X_train]  # stack: n_epochs, n_channels, n_times
    y_train = np.empty([0])
    for i in range(len(X_train)):
        y_train = np.hstack((y_train, (i+1)*np.ones(X_train[i].shape[0]).astype(int)))  # targets
        print('Number of samples in training class %i : %i' % (i+1, X_train[i].shape[0]))
    #print(y_train, y_train.shape)

    X_train = np.concatenate(X_train, axis=0)

    if query_classes_test is not None:
        X_test = [np.concatenate(d, axis=1) for d in X_test] # signals: n_epochs, n_channels, n_times
        y_test = np.empty([0])
        for i in range(len(X_test)):
            y_test = np.hstack((y_test, (i+1)*np.ones(X_test[i].shape[0]).astype(int)))  # targets
            print('Number of samples in test class %i : %i' % (i+1, X_test[i].shape[0]))
        X_test = np.concatenate(X_test, axis=0)
    else:
        y_test = None

    data = {}
    data['times'] = curr_epochs_query.times
    data['X_train'] = X_train
    data['X_test'] = X_test
    data['y_train'] = y_train
    data['y_test'] = y_test

    del X_train, X_test, y_train, y_test, curr_epochs
    if k > 1:
        data = cat_subsequent_timepoints(k, data)
    
    return data

def cat_subsequent_timepoints(k, data):
    '''
    :param k: (scalar) number of subsequent time points
    :param data: (dict) has the following keys -
           times: n_times
           X_train: n_epochs, n_channels, n_times
           y_train: n_epochs
           X_test: n_epochs, n_channels, n_times
           y_test: n_epochs
    :return:
    new_times = floor(n_times/k)
    new_X_train: n_epochs, n_channels * k, floor(n_times/k)
    new_X_test: n_epochs, n_channels * k, floor(n_times/k)
    '''

    n_epochs, n_channels, n_times = data['X_train'].shape
    n_times_round = int(k*np.floor(n_times/k)) # remove residual mod k
    assert n_times_round > 0
    data['X_train'] = data['X_train'][:,:,0:n_times_round]

    new_data = data.copy()
    new_data['times'] = data['times'][0:n_times_round:k]
    new_data['X_train'] = data['X_train'].reshape((n_epochs, -1, int(n_times_round/k)), order='F')

    if data['X_test'] is not None:
        n_epochs_test = data['X_test'].shape[0]
        data['X_test'] = data['X_test'][:, :, 0:n_times_round]
        new_data['X_test'] = data['X_test'].reshape((n_epochs_test, -1, int(n_times_round/k)), order='F')


    return new_data


def train_test_GAT(data):
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from mne.decoding import (GeneralizingEstimator, Scaler, cross_val_multiscore, LinearModel, get_coef, Vectorizer)

    # Define a classifier for GAT
    #clf = make_pipeline(StandardScaler(), LinearSVC())
    if max(data['y_train']) > 2:
        print('Multiclass classification')
        clf = make_pipeline(StandardScaler(), OneVsRestClassifier(LogisticRegression(solver='lbfgs')))
    else:
        print('Binary classification')
        clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(solver='lbfgs')))

    # Define the Temporal Generalization object
    time_gen = GeneralizingEstimator(clf, n_jobs=1, scoring='roc_auc', verbose=True)
    # Fit model
    if (data['X_test'] is not None) and (data['y_test'] is not None): # Generalization across conditions
        #print(X_train, y_train, X_test, y_test)
        print('X_train, y_train, shapes:')
        print(data['X_train'].shape, data['y_train'].shape)
        time_gen.fit(data['X_train'], data['y_train'])
        scores = time_gen.score(data['X_test'], data['y_test'])
        scores = np.expand_dims(scores, axis=0) # For later compatability (plot_GAT() np.mean(scores, axis=0))
        #print(scores)
    else: # Generlization across time only (not across conditions or modalities)
        scores = cross_val_multiscore(time_gen, data['X_train'], data['y_train'], cv=5, n_jobs=1)

    return time_gen, scores


def plot_GAT(times, time_gen, scores):
    # Plot the diagonal
    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)
    fig1, ax = plt.subplots()
    ax.plot(times, np.diag(scores), label='score')
    ax.axhline(.5, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('AUC')
    ax.legend()
    ax.axhline(.5, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('AUC')
    ax.set_ylim(0.4, np.max([0.8, np.max(np.diag(scores))]))
    ax.legend()
    ax.axvline(.0, color='k', linestyle='-')
    ax.set_title('Decoding over time')

    # Plot the full GAT matrix
    fig2, ax = plt.subplots(1, 1)
    im = ax.imshow(scores, interpolation='lanczos', origin='lower', cmap='Reds',
                   extent=times[[0, -1, 0, -1]], vmin=0.5, vmax=np.max([0.8, np.max(scores)]))
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Temporal Generalization')
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')
    plt.colorbar(im, ax=ax)

    return fig1, fig2

def filter_relevant_epochs_filenames(filenames, picks):
    '''
    returns filenames that match the criteria in picks, which could be either:
    - picks: a list of strings or integers. If strings then regardered as ROIs. If 'all' in picks then all filenames will be inculded. If 'none' is the only string in the list then no filenames will be picked (since no ROI is called 'none'). If integers then regarded as channel numbers to be picked. 
    - channel numbers (list of int)
    - ROIs (list of strings), e.g., ['RSTG', 'LSTG']
    - 'all' (string)
    '''
    #print(picks)
    if all(isinstance(s, str) for s in picks): # check if ROIs
        filtered_filenames = []
        for fn in filenames:
            bn = os.path.basename(fn)
            ugly_hack = 0
            if 'patient_479_11' in bn or 'patient_479_25' in bn:
                ugly_hack = 1
            probe_name = bn.split('_')[3+ugly_hack]
            if (probe_name in picks) | ('all' in picks):
                filtered_filenames.append(fn)

    elif all(isinstance(i, int) for i in picks): # Check if a list of channels
        filtered_filenames = [fn for c, fn in zip(filenames, channels) if c in pick_micro]
    else:
        raise('Yair: Type error of picks')

    return filtered_filenames


def get_train_test_data_from_epochs(epochs, queries_train, queries_test, X_train, X_test, train_times):
    ''' Append to X_train and X_test new epochs based on current train and test queries
    '''
    for q, query_train in enumerate(queries_train):
        epochs_train = epochs[query_train]
        epochs_train.crop(train_times["start"], train_times["stop"])
        X_train[q].append(epochs_train._data)
        print('Train epochs num_epochs X num_channels X num_timepoints:', epochs_train._data.shape)
        
    if queries_test is not None:
        for q, query_test in enumerate(queries_test):
            epochs_test = epochs[query_test]
            epochs_test.crop(train_times["start"], train_times["stop"])
            X_test[q].append(epochs_test._data)
    else: # no test queries (generalization across time only, not conditions)
        X_test = None

    return X_train, X_test, epochs_train
           

def prepare_data_for_classifier(epochs_list, queries,
                                list_class_numbers=None,
                                min_trials = 0):
    '''
    cat epochs data across channel dimension and then prepare for classifier
    '''
    X_all_queries = []; y_all_queries = []; stimuli = []
    for q, query in enumerate(queries):
        if 'heard' in query: # HACK! for word_string
            continue
        if 'END_OF_WAV' in query: # HACK! for phone_string
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
        X = np.concatenate(X, axis=1) # cat along channel (feature) dimension
        X_all_queries.append(X)
        # common y vector
        num_trials = curr_data.shape[0]
        #print(q, query, num_trials)
        if list_class_numbers:
            class_number = list_class_numbers[q]
        else:
            class_number = q + 1
        y_all_queries.append(np.full(num_trials, class_number))

    X_all_queries = np.concatenate(X_all_queries, axis=0) # cat along the trial dimension
    y_all_queries = np.concatenate(y_all_queries, axis=0)
    
    return X_all_queries, y_all_queries, np.asarray(stimuli)

#if c==0 and i==0:
    #print(epochs_class_test.metadata['sentence_string'])
    #print(', '.join(epochs_class_test.metadata['word_string']))
