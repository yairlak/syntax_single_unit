
import argparse, os, sys
import mne
from utils.data_manip import DataHandler
from utils import classification, comparisons, load_settings_params, data_manip
from utils.utils import dict2filename, update_queries, probename2picks, pick_responsive_channels
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from mne.decoding import (cross_val_multiscore, LinearModel, GeneralizingEstimator)
import numpy as np
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.multiclass import OneVsRestClassifier

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(description='Generate plots for TIMIT experiment')
# DATA
parser.add_argument('--patient', action='append', default=[], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], action='append', default=[], help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default=None, help='')
parser.add_argument('--filter', choices=['raw', 'high-gamma'], action='append', default=[], help='')
parser.add_argument('--probe-name', default=[], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
# QUERY
parser.add_argument('--comparison-name', default=None, help='Comparison name from Code/Main/functions/comparisons.py')
parser.add_argument('--comparison-name-test', default=None, help='Comparison name from Code/Main/functions/comparisons.py')
parser.add_argument('--block-train', choices=['auditory', 'visual'], default='visual', help='Block type will be added to the query in the comparison')
parser.add_argument('--block-test', choices=['auditory', 'visual'], default=None, help='Block type will be added to the query in the comparison')
parser.add_argument('--fixed-constraint', default=None, help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--fixed-constraint-test', default=None, help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--classifier', default='ridge', choices=['svc', 'logistic', 'ridge'], help='Specify a classifier type to be used')
parser.add_argument('--min-trials', default=10, type=float, help='Minimum number of trials from each class.')
# MISC
parser.add_argument('--tmin', default=None, type=float, help='crop window. If empty, only crops 0.1s from both sides, due to edge effects.')
parser.add_argument('--tmax', default=None, type=float, help='crop window')
parser.add_argument('--vmin', default=None, type=float, help='')
parser.add_argument('--vmax', default=None, type=float, help='')
parser.add_argument('--decimate', default=None, type=float, help='If not empty, (for speed) decimate data by the provided factor.')
parser.add_argument('--cat-k-timepoints', type=int, default=1, help='How many time points to concatenate before classification')
parser.add_argument('--path2figures', default='../../Figures/Decoding', help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--dont-overwrite', default=False, action='store_true', help="If True then file will be overwritten")
# PARSE
args = parser.parse_args()
args.patient = ['patient_' + p for p in  args.patient]
print(mne.__version__)

if args.comparison_name == args.comparison_name_test: args.comparison_name_test = None
if args.block_train == args.block_test: args.block_test = None
# Which args to have in fig filename
list_args2fname = ['patient', 'data_type', 'filter', 'comparison_name', 'block_train']
if args.block_test: list_args2fname += ['comparison_name_test', 'block_test']
if args.responsive_channels_only: list_args2fname += ['responsive_channels_only']
if args.probe_name: list_args2fname.append('probe_name')
print(list_args2fname)

######################
# Queries TRAIN/TEST #
######################

# COMPARISON
comparisons = comparisons.comparison_list()
comparison = comparisons[args.comparison_name].copy()

if 'level' in comparison.keys():
    args.level = comparison['level']

print(args)
#############
# LOAD DATA #
#############
data = DataHandler(args.patient, args.data_type, args.filter,
                   args.probe_name, args.channel_name, args.channel_num)
# Both neural and feature data into a single raw object
data.load_raw_data(decimate=args.decimate)
data.epoch_data(level=args.level,
                scale_epochs=False,
                verbose=True)


assert data.epochs
print('Channel names:')
[print(e.ch_names) for e in data.epochs]

comparison = update_queries(comparison, args.block_train, args.fixed_constraint, data.epochs[0].metadata)
print(comparison)
# Generalization to TEST SET
if args.block_test:
    if not args.comparison_name_test: # Generalization across modalities only
        args.comparison_name_test = args.comparison_name 
    comparison_test = comparisons[args.comparison_name_test].copy()
    comparison_test = update_queries(comparison_test, args.block_test, args.fixed_constraint_test, data.epochs[0].metadata)
    print(comparison_test)
else:
    if args.comparison_name_test:
        raise('block-test and comparison-name-test must be specified jointly.')

###########################
# Classification pipeline #
###########################


def prepare_data_for_classifier(epochs, queries):
    '''
    '''
# GET SENTENCE-LEVEL DATA BEFORE SPLIT
    X = []; y = []; stimuli = []
    for q, query in enumerate(queries):
        if 'heard' in query: # HACK! for word_string
            continue
        if 'END_OF_WAV' in query: # HACK! for phone_string
            continue
        if epochs[query].get_data().shape[0] < args.min_trials:
            print(f'Less than {args.min_trials} trials matched query: {query}')
            continue

        stimuli.append(epochs[query].metadata[['sentence_string', 'word_string']])
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
        clf = make_pipeline(OneVsRestClassifier(LogisticRegression(solver='lbfgs', class_weight='balanced')))
    elif args.classifier == 'svc':
        clf = make_pipeline(OneVsRestClassifier(LinearSVC(class_weight='balanced')))
    elif args.classifier == 'ridge':
        clf = make_pipeline(OneVsRestClassifier(RidgeClassifier(class_weight='balanced')))
    time_gen = GeneralizingEstimator(clf, scoring='roc_auc_ovo', n_jobs=1)
else:
    print('Binary classification')
    if args.classifier == 'logistic':
        clf = make_pipeline(LinearModel(LogisticRegression(C=1, solver='liblinear', class_weight='balanced')))
    elif args.classifier == 'svc':
        clf = make_pipeline(LinearSVC(class_weight='balanced'))
    elif args.classifier == 'ridge':
        clf = make_pipeline(LinearModel(RidgeClassifier(class_weight='balanced')))
    time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=1)

# PREPARE DATA
X_train_list, X_test_list, X_list = ([],[],[])
for i_epochs, epochs in enumerate(data.epochs):
    if args.comparison_name_test: # Generalization across modalities/conditions
        X_train, y_train, stimuli_train = prepare_data_for_classifier(epochs, comparison['queries']) 
        print('Train epochs num_epochs X num_channels X num_timepoints:', X_train.shape, y_train.shape)
        X_test, y_test, stimuli_test = prepare_data_for_classifier(epochs, comparison_test['queries']) 
        print('Train epochs num_epochs X num_channels X num_timepoints:', X_test.shape, y_test.shape)
        X_train_list.append(X_train); X_test_list.append(X_test)
    else:
        X, y, stimuli_train = prepare_data_for_classifier(epochs, comparison['queries'])
        X_list.append(X)
        print(X.shape, y.shape, np.min(X), np.max(X), np.min(y), np.max(y))
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
else: # Fit and eval with CV over X, y
    X = np.concatenate(X_list, axis=1)
    scores = cross_val_multiscore(time_gen, X, y, cv=5, n_jobs=1)

############
# PLOTTING #
############
fig, ax = plt.subplots(1)
vmax = np.max(np.mean(scores, axis=0))
im = ax.matshow(np.mean(scores, axis=0), cmap='RdBu_r', origin='lower', extent=epochs.times[[0, -1, 0, -1]], vmin=1-vmax, vmax=vmax)
ax.axhline(0., color='k')
ax.axvline(0., color='k')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(np.arange(epochs.tmin, epochs.tmax, 0.2))
ax.set_yticks(np.arange(epochs.tmin, epochs.tmax, 0.2))
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title(f'{args.comparison_name} {args.block_train} {args.comparison_name_test} {args.block_test}')
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
