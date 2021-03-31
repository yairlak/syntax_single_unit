import argparse, os, sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
import mne
from functions import classification, comparisons, load_settings_params
from functions.utils import dict2filename, update_queries, probename2picks, pick_responsive_channels
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from mne.decoding import (cross_val_multiscore, LinearModel, GeneralizingEstimator)
import numpy as np
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from pprint import pprint
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.multiclass import OneVsRestClassifier

parser = argparse.ArgumentParser(description='Generate plots for TIMIT experiment')
# DATA
parser.add_argument('--patient', action='append', default=[], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], action='append', default=[], help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='word', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'high-gamma'], action='append', default=[], help='')
parser.add_argument('--probe-name', default=[], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
# QUERY
parser.add_argument('--comparison-name', default='first_last_word', help='Comparison name from Code/Main/functions/comparisons.py')
parser.add_argument('--comparison-name-test', default=[], help='Comparison name from Code/Main/functions/comparisons.py')
parser.add_argument('--block-type', choices=['auditory', 'visual'], default='auditory', help='Block type will be added to the query in the comparison')
parser.add_argument('--block-type-test', choices=['auditory', 'visual', []], default=[], help='Block type will be added to the query in the comparison')
parser.add_argument('--fixed-constraint', default=[], help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--fixed-constraint-test', default=[], help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--classifier', default='logistic', choices=['svc', 'logistic', 'ridge'], help='Specify a classifier type to be used')
# MISC
parser.add_argument('--tmin', default=[], type=float, help='crop window. If empty, only crops 0.1s from both sides, due to edge effects.')
parser.add_argument('--tmax', default=[], type=float, help='crop window')
parser.add_argument('--vmin', default=None, type=float, help='')
parser.add_argument('--vmax', default=None, type=float, help='')
parser.add_argument('--decimate', default=[], type=float, help='If not empty, (for speed) decimate data by the provided factor.')
parser.add_argument('--cat-k-timepoints', type=int, default=1, help='How many time points to concatenate before classification')
parser.add_argument('--path2figures', default=[], help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--dont-overwrite', default=False, action='store_true', help="If True then file will be overwritten")
# PARSE
args = parser.parse_args()
patients = ['patient_' + p for p in  args.patient]
print(mne.__version__)

# Which args to have in fig filename
list_args2fname = ['patient', 'data_type', 'filter', 'level', 'comparison_name', 'block_type']
if args.block_type_test: list_args2fname += ['comparison_name_test', 'block_type_test']
if args.responsive_channels_only: list_args2fname += ['responsive_channels_only']
if args.probe_name: list_args2fname.append('probe_name')
print(list_args2fname)

if not args.path2figures:
    args.path2figures = os.path.join('..', '..', 'Figures', 'Decoding')
print(args)

########
# DATA #
########

# LOAD
epochs_list = []
for p, (patient, data_type, filt, probe_name) in enumerate(zip(patients, args.data_type, args.filter, args.probe_name)):
    print(patient, data_type, filt, probe_name)
    try:
        settings = load_settings_params.Settings(patient)
        fname = '%s_%s_%s_%s-epo.fif' % (patient, data_type, filt, args.level)
        epochs = mne.read_epochs(os.path.join(settings.path2epoch_data, fname), preload=True)
    except:
        print(f'WARNING: data not found for {patient} {data_type} {filt} {args.level}')
        continue

    # PICK
    if args.probe_name:
        picks = probename2picks(probe_name, epochs.ch_names, data_type)
    elif args.channel_name: # channel_name OVERWRITES probe_name if not empty
        picks = args.channel_name
    elif args.channel_num:
        picks = args.channel_num
    print(picks)
    if (not picks) and (picks is not None): # check if empty list (but not None). If yes, continue. If picks is None then takes all channels.
        continue
    epochs.pick(picks)

    # Filter non-responsive channels
    if args.responsive_channels_only:
        if args.block_type_test: # pick significant channels from both modalities
            block_types = list(set([args.block_type, args.block_type_test]))
        else:
            block_types = [args.block_type] # In next line, list is expected with all block types 
        picks = pick_responsive_channels(epochs.ch_names, patient, data_type, filt, block_types , p_value=0.05)
        if picks:
            epochs.pick_channels(picks)
        else:
            print('WARNING: No responsive channels were found')
            continue

    # CROP
    if args.tmin and args.tmax: epochs.crop(args.tmin, args.tmax)

    # DECIMATE
    if args.decimate: epochs.decimate(args.decimate)

    epochs_list.append(epochs)

[print(e.ch_names) for e in epochs_list]

######################
# Queries TRAIN/TEST #
######################

# COMPARISON
comparisons = comparisons.comparison_list()
comparison = comparisons[args.comparison_name].copy()
comparison = update_queries(comparison, args.block_type, args.fixed_constraint, epochs.metadata)

# Generalization to TEST SET
if args.comparison_name_test:
    comparison_test = comparisons[args.comparison_name_test].copy()
    if not args.block_type_test:
        raise('block-type-test must be specified if comparison-name-test is provided')
    comparison_test = update_queries(comparison_test, args.block_type_test, args.fixed_constraint_test, epochs.metadata)


###########################
# Classification pipeline #
###########################

# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 

def prepare_data_for_classifier(epochs, queries):
    '''
    '''
    X = []; y = []; stimuli = []
    for q, query in enumerate(queries):
        stimuli.append(epochs[query].metadata['sentence_string'])
        curr_data = epochs[query].get_data()
        X.append(curr_data)
        num_trials = curr_data.shape[0]
        y.append(np.full(num_trials, q))
     
    X = np.concatenate(X, axis=0) # cat along the trails dimension
    y = np.concatenate(y, axis=0)
    
    return X, y, stimuli

# Prepare pipeline and set classifier
if len(comparison['queries']) > 2:
    print('Multiclass classification')
    if args.classifier == 'logistic':
        clf = make_pipeline(StandardScaler(), OneVsRestClassifier(LogisticRegression(solver='lbfgs')))
    elif args.classifier == 'ridge':
        clf = make_pipeline(StandardScaler(), OneVsRestClassifier(RidgeClassifier()))
    time_gen = GeneralizingEstimator(clf, scoring='roc_auc_ovo', n_jobs=1)
else:
    print('Binary classification')
    if args.classifier == 'logistic':
        clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(C=1, solver='liblinear')))
    elif args.classifier == 'ridge':
        clf = make_pipeline(StandardScaler(), LinearModel(RidgeClassifier()))
    elif args.classifier == 'lstm':
        clf = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
    time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=1)

# PREPARE DATA
X_train_list, X_test_list, X_list = ([],[],[])
for i_epochs, epochs in enumerate(epochs_list):
    if args.comparison_name_test: # Generalization across modalities/conditions
        X_train, y_train, stimuli_train = prepare_data_for_classifier(epochs, comparison['queries']) 
        print('Train epochs num_epochs X num_channels X num_timepoints:', X_train.shape, y_train.shape)
        X_test, y_test, stimuli_test = prepare_data_for_classifier(epochs, comparison_test['queries']) 
        print('Train epochs num_epochs X num_channels X num_timepoints:', X_test.shape, y_test.shape)
        X_train_list.append(X_train); X_test_list.append(X_test)
    else:
        X, y, stimuli_train = prepare_data_for_classifier(epochs, comparison['queries'])
        X_list.append(X)
        print(X.shape, y.shape)
    if i_epochs == 0:
        [print(df) for df in stimuli_train]
        if args.comparison_name_test:
            [print(df) for df in stimuli_test]

# TRAIN AND EVAL
if args.comparison_name_test: # Generalization across modalities/conditions
    X_train = np.concatenate(X_train_list, axis=1)
    X_test = np.concatenate(X_test_list, axis=1)
    time_gen.fit(X_train, y_test)
    scores = time_gen.score(X_test, y_test)
    scores = np.expand_dims(scores, axis=0) # For later compatability: add singelton for splits, for averaging across: np.mean(scores, axis=0)
else:
    X = np.concatenate(X_list, axis=1)
    scores = cross_val_multiscore(time_gen, X, y, cv=5, n_jobs=1)

############
# PLOTTING #
############
fig, ax = plt.subplots(1)
im = ax.matshow(np.mean(scores, axis=0), cmap='RdBu_r', origin='lower', extent=epochs.times[[0, -1, 0, -1]], vmin=args.vmin, vmax=args.vmax)
ax.axhline(0., color='k')
ax.axvline(0., color='k')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(np.arange(epochs.tmin, epochs.tmax, 0.2))
ax.set_yticks(np.arange(epochs.tmin, epochs.tmax, 0.2))
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title(f'{args.comparison_name} {args.block_type} {args.comparison_name_test} {args.block_type_test}')
plt.colorbar(im, ax=ax)


########
# SAVE #
########
if len(list(set(args.data_type))) == 1: args.data_type = list(set(args.data_type))
if len(list(set(args.filter))) == 1: args.filter = list(set(args.filter))
args.probe_name = sorted(list(set([item for sublist in args.probe_name for item in sublist]))) # !! lump together all probe names !! to reduce filename length
print(args.__dict__, list_args2fname)
fname_fig = dict2filename(args.__dict__, '_', list_args2fname, 'png', True)
fname_fig = os.path.join(args.path2figures, 'GAT_' + fname_fig)
fig.savefig(fname_fig)
print('Figures saved to: ' + fname_fig)
