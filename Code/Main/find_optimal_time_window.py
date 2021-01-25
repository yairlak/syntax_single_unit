import argparse, os, sys, pickle
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
import mne
from mne.decoding import (cross_val_multiscore, LinearModel, GeneralizingEstimator)
from functions import classification, comparisons, load_settings_params
from functions.utils import dict2filename, update_queries, probename2picks, pick_responsive_channels
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.manifold import MDS
from sklearn.model_selection import ShuffleSplit
from sklearn.cluster import AgglomerativeClustering
from skorch import NeuralNetClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from scipy import stats
from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import torch
import models # custom module with neural-network models (LSTM/GRU/CNN)
import pandas as pd
from statsmodels.multivariate.manova import MANOVA

parser = argparse.ArgumentParser(description='Generate plots for TIMIT experiment')
# DATA
parser.add_argument('--patient', action='append', default=[], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], action='append', default=[], help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='word', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'high-gamma'], action='append', default=[], help='')
parser.add_argument('--probe-name', default=[], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channe-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
# QUERY
parser.add_argument('--comparison-name', default='first_last_word', help='Comparison name from Code/Main/functions/comparisons.py')
parser.add_argument('--comparison-name-test', default=[], help='Comparison name from Code/Main/functions/comparisons.py')
parser.add_argument('--block-type', choices=['auditory', 'visual'], default='auditory', help='Block type will be added to the query in the comparison')
parser.add_argument('--block-type-test', choices=['auditory', 'visual', []], default=[], help='Block type will be added to the query in the comparison')
parser.add_argument('--fixed-constraint', default=[], help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--fixed-constraint-test', default=[], help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--label-from-metadata', default=[], help='Field name in metadata that will be used to generate labels for the different classes. If empty, condition_names in comparison will be used')
parser.add_argument('--pick-classes', default=[], type=str, nargs='*', help='Limit the classes to this list')
# MODEL
parser.add_argument('--model-type', default='lstm', choices=['logistic', 'lstm', 'cnn']) # 'svc' and 'ridge' are omited since they don't implemnent predict_proba (although there's a work around, using their decision function and map is to probs with eg softmax)
parser.add_argument('--cuda', default=False, action='store_true', help="If True then file will be overwritten")
# MISC
parser.add_argument('--vmin', default=None, type=float, help='')
parser.add_argument('--vmax', default=None, type=float, help='')
parser.add_argument('--times', nargs='*', default=[0.1], type=float, help='')
parser.add_argument('--time-window', default=0.5, type=float, help='')
parser.add_argument('--num-bins', default=[], type=int, help='')
parser.add_argument('--min-trials', default=15, type=float, help='Minimum number of trials from each class.')
parser.add_argument('--decimate', default=[], type=float, help='If not empty, (for speed) decimate data by the provided factor.')
parser.add_argument('--path2figures', default=[], help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--dont-overwrite', default=False, action='store_true', help="If True then file will be overwritten")
parser.add_argument('--path2output', default=[], help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
# PARSE
args = parser.parse_args()
patients = ['patient_' + p for p in  args.patient]
print(mne.__version__)

########
# INIT #
########
USE_CUDA = args.cuda  # Set this to False if you don't want to use CUDA

# SET SEEDS
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

# Which args to have in fig filename
list_args2fname = ['patient', 'data_type', 'filter', 'level', 'comparison_name', 'block_type', 'time_window', 'num_bins', 'min_trials']
if args.block_type_test: list_args2fname += ['comparison_name_test', 'block_type_test']
if args.probe_name: list_args2fname.append('probe_name')
if args.responsive_channels_only: list_args2fname += ['responsive_channels_only']
print('args2fname', list_args2fname)

if not args.path2figures:
    args.path2figures = os.path.join('..', '..', 'Figures', 'RSA')
if not args.path2output:
    args.path2output = os.path.join('..', '..', 'Output', 'RSA')
print('args\n', args)


########
# DATA #
########

# LOAD
epochs_list = []
for p, (patient, data_type, filt, probe_name) in enumerate(zip(patients, args.data_type, args.filter, args.probe_name)):
    try:
        settings = load_settings_params.Settings(patient)
        fname = '%s_%s_%s_%s-epo.fif' % (patient, data_type, filt, args.level)
        epochs = mne.read_epochs(os.path.join(settings.path2epoch_data, fname), preload=True)
    except: # Exception:
        print(f'WARNING: data not found for {patient} {data_type} {filt} {args.level}')
        continue

    # PICK
    if args.probe_name:
        picks = probename2picks(probe_name, epochs.ch_names, data_type)
    elif args.channel_name:
        picks = args.channel_name
    elif args.channel:
        picks = args.channel_num
    print('picks:', picks)
    if not picks:
        continue

    epochs.pick(picks)

    # RESPONSIVE CHANNELS
    if args.responsive_channels_only:
        if args.block_type_test: # pick significant channels from both modalities
            block_types = list(set([args.block_type, args.block_type_test]))
        else:
            block_types = [args.block_type] # In next line, list is expected with all block types
        picks = pick_responsive_channels(epochs.ch_names, patient, data_type, filt, block_types , p_value=0.05)
        if picks:
            epochs.pick_channels(picks)
        else:
            print(f'WARNING: No responsive channels were found for {patient} {data_type} {filt} {probe_name}')
            continue


    # DECIMATE
    if args.decimate: epochs.decimate(args.decimate)
    
    metadata = epochs.metadata
    metadata['word_start'] = metadata.apply(lambda row: row['word_string'][0], axis=1)
    metadata['word_end'] = metadata.apply(lambda row: row['word_string'][-1], axis=1)
    epochs.metadata = metadata
    epochs_list.append(epochs)

print('Channel names:')
[print(e.ch_names) for e in epochs_list]
######################
# Queries TRAIN/TEST #
######################

# COMPARISON
comparisons = comparisons.comparison_list()
comparison = comparisons[args.comparison_name].copy()
comparison = update_queries(comparison, args.block_type, args.fixed_constraint, epochs.metadata)
print('Comparison:')
print(comparison)

# Generalization to TEST SET
if args.comparison_name_test:
    comparison_test = comparisons[args.comparison_name_test].copy()
    if not args.block_type_test:
        raise('block-type-test must be specified if comparison-name-test is provided')
    comparison_test = update_queries(comparison_test, args.block_type_test, args.fixed_constraint_test)


def prepare_data_for_classifier(epochs, comparison, pick_classes, field_for_labels=[]):
    '''
    '''
    X = []; y = []; labels = []; cnt = 0
    for q, query in enumerate(comparison['queries']):
        if 'heard' in query: # HACK! for word_string
            continue
        if 'END_OF_WAV' in query: # HACK! for phone_string
            continue
        if epochs[query].get_data().shape[0] < args.min_trials:
            #print(f'Less than {args.min_trials} trials matched query: {query}')
            continue
        if field_for_labels: # add each value of a feature as a label (e.g., for word_length - 2, 3, 4..)
            label = list(set(epochs[query].metadata[field_for_labels]))
            assert len(label) == 1
            label = label[0]
        else:
            label = comparison['condition_names'][q]
        if pick_classes and (label not in pick_classes):
            continue
        labels.append(label)
        curr_data = epochs[query].get_data()
        X.append(curr_data)
        num_trials = curr_data.shape[0]
        y.append(np.full(num_trials, cnt))
        cnt += 1

    X = np.concatenate(X, axis=0) # cat along the trial dimension
    y = np.concatenate(y, axis=0)

    return X, y, labels

if args.num_bins:
    bin_size = args.time_window / args.num_bins
for t in args.times:
    # PREPARE DATA
    X_list = []
    for epochs in epochs_list: # loop over epochs from different patients or probes
        ###############
        # BINNIZATION #
        ###############
        if args.num_bins:
            X = []
            for i_bin in range(args.num_bins):
                print(i_bin)
                curr_epochs = epochs.copy().crop(t+i_bin*bin_size, t+(i_bin+1)*bin_size)
                curr_X, y, labels = prepare_data_for_classifier(curr_epochs, comparison, args.pick_classes, args.label_from_metadata)
                curr_X = np.mean(curr_X, axis=2, keepdims=True) # curr_X: (num_trials X num_channels X 1)
                X.append(curr_X)
            X = np.concatenate(X, axis=2) # X: num_trials x num_channels, num_bins
        else:
            X, y, labels = prepare_data_for_classifier(epochs.copy().crop(t, t+args.time_window), comparison, args.pick_classes, args.label_from_metadata)

        if not labels:
            labels = comparison['condition_names']
        X_list.append(X)
        #print('Shapes (X, y): ', X.shape, y.shape)
    X = np.concatenate(X_list, axis=1) # cat different patients/probes as new channel features (num_trials x num_channels, num_bins)
  
    classes = list(set(y))
    num_classes=len(classes)
    
    df = pd.DataFrame(np.hstack((X.reshape(X.shape[0], -1, order='C'), y))) # first dim changes the slowest during reading/writing; num_trials in this case
    print(classes, labels)
    print('Shapes (X, y): ', X.shape, y.shape)
    print(df)
