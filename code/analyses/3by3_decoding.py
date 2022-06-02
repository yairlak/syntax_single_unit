import argparse, os
import mne
from decoding.utils import update_args
from decoding.data_manip import get_data
from utils import data_manip, classification, comparisons
from utils.utils import dict2filename, update_queries, probename2picks
from utils.classification import prepare_data_for_classifier
from utils.data_manip import DataHandler
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
from decoding.data_manip import get_3by3_train_test_data, lump_data_together

parser = argparse.ArgumentParser(description='Generate plots for TIMIT experiment')
# DATA
parser.add_argument('--patient', action='append',
                    default=[], type=str, help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'],
                    action='append', default=[], help='electrode type')
parser.add_argument('--filter', choices=['raw','high-gamma'],
                    action='append', default=[], help='')
parser.add_argument('--level',
                    choices=['sentence_onset','sentence_offset',
                             'word', 'phone'],
                    default='phone', help='')
parser.add_argument('--smooth', default=None, type=int, help='')
parser.add_argument('--ROIs', default=None, nargs='*', type=str,
                    help='e.g., Brodmann.22-lh, overrides probe_name')
parser.add_argument('--probe-name', default=[], nargs='*',
                    action='append', type=str,
                    help='Probe name to plot (ignore channel-name/num)')
parser.add_argument('--channel-name', default=[], nargs='*', action='append',
                    type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=[], nargs='*', action='append',
                    type=int,
                    help='channel number (if empty list [] then all channels)')
parser.add_argument('--responsive-channels-only', action='store_true',
                    default=False,
                    help='Include only responsive channels.')
parser.add_argument('--data-type_filters',
                    choices=['micro_high-gamma','macro_high-gamma',
                             'micro_raw','macro_raw', 'spike_raw'], nargs='*',
                             default=[], help='Only if args.ROIs is used')
# QUERY
parser.add_argument('--comparison-name', default=None,
                    help='See Utils/comparisons.py')
parser.add_argument('--comparison-name-test', default=None,
                    help='See Utils/comparisons.py')
parser.add_argument('--block-train', choices=['auditory', 'visual'],
                    default='auditory',
                    help='Block type is added to the query in the comparison')
parser.add_argument('--block-test', choices=['auditory', 'visual'],
                    default=None,
                    help='Block type is added to the query in the comparison')
parser.add_argument('--fixed-constraint', default=None,
                    help='e.g., "and first_phone == 1"')
parser.add_argument('--fixed-constraint-test', default=None,
                    help='e.g., "and first_phone == 1"')
parser.add_argument('--classifier', default='logistic',
                    choices=['svc', 'logistic', 'ridge'])
parser.add_argument('--min-trials', default=10, type=float,
                    help='Minimum number of trials from each class.')
parser.add_argument('--n-splits', default=2, type=int,
                    help='Number of CV splits.')
parser.add_argument('--equalize-classes', default='downsample',
                    choices=['upsample', 'downsample'])

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
args = update_args(args)

args.query = ''
print(mne.__version__)

# Which args to have in fig filename
list_args2fname = ['patient', 'data_type', 'filter', 'level']
if args.responsive_channels_only: list_args2fname += ['responsive_channels_only']
if args.probe_name: list_args2fname.append('probe_name_lumpped')
print(list_args2fname)

if not args.path2figures:
    args.path2figures = os.path.join('..', '..', 'Figures', 'Decoding')
print(args)

########
# DATA #
########
data = get_data(args)
# GET SENTENCE-LEVEL DATA BEFORE SPLIT
data.epoch_data(level=args.level,
                query=None,
                scale_epochs=False,
                smooth=args.smooth,
                verbose=True)

print(data.epochs[0])
print('Channel names:')
[print(e.ch_names) for e in data.epochs]

###########
# STIMULI #
###########
phones = {}
phones['place'] = {}
phones['place']['nasals'] = ['M', 'N', 'NG'] # Classification between nasals with different place features.
phones['place']['voiced_plosives'] = ['B', 'D', 'G'] # between voiced_plosives with different place features
phones['place']['voiceless_plosives'] = ['P', 'T', 'K'] # etc
phones['manner'] = {}
phones['manner']['labial'] = ['M', 'B', 'P'] # Classification between labials with different manners
phones['manner']['coronal'] = ['N', 'D', 'T'] # etc
phones['manner']['dorsal'] = ['NG', 'G', 'K'] # etc
phone_strings = ['M', 'N', 'NG', 'B', 'D', 'G', 'P', 'T', 'K']

#########################
# TRAIN AND EVAL MODELS #
#########################

# CHOOSE CLASIFIER
if args.classifier == 'logistic':
    clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(C=1, solver='liblinear', class_weight='balanced')))
elif args.classifier == 'svc':
    clf = make_pipeline(StandardScaler(), LinearSVC(class_weight='balanced'))
elif args.classifier == 'ridge':
    clf = make_pipeline(StandardScaler(), LinearModel(RidgeClassifier(class_weight='balanced')))
# GAT
time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=-1)

data_phones = get_3by3_train_test_data(data.epochs,
                                       phone_strings,
                                       args.n_splits,
                                       args)
# RUNNING EXAMPLE: trained on (M vs (N, NG)), tested on (B vs (D, G)):
models = {}  # NESTED DICTS: models[classify_dimension][train_at_feature][ph][CV-split]
scores = {}  # NESTED DICTS: scores[classify_dimension][train_at_feature][ph][CV-split][test_at_feature]
for classify_dim in ['manner', 'place']:
    print('\n', '-'*100)
    print(f'Classify among different {classify_dim.upper()} of articulation')
    print('-'*100, '\n') 
    feature_names = phones[classify_dim].keys()
    models[classify_dim] = {}
    scores[classify_dim] = {}
    for i_split in range(args.n_splits):
        print(f'Split - {i_split+1}')
        models[classify_dim][i_split] = {}
        scores[classify_dim][i_split] = {}
        for train_at_feature in feature_names: # fixed feature (e.g., nasals with different place features)
            models[classify_dim][i_split][train_at_feature] = {}
            scores[classify_dim][i_split][train_at_feature] = {}
            phones_train = phones[classify_dim][train_at_feature]
            for i_ph_train, ph_train in enumerate(phones_train): # which phone in the triplet (e.g., /M/ vs (/N/ and /NG/))
                models[classify_dim][i_split][train_at_feature][i_ph_train] = {}
                scores[classify_dim][i_split][train_at_feature][i_ph_train] = {}
                # GET DATA FOR TRAIN FEATURE (e.g., nasals: M vs (N, NG))
                other_phs_train = [ph for ph in phones_train if ph !=  ph_train]            
                X_train, y_train, stimuli_train = \
                    lump_data_together(data_phones,
                                       ph_train, other_phs_train,
                                       i_split, 'train')
                print('-'*100)
                print(f'\nTrain model at {train_at_feature} ({X_train.shape}, {y_train.shape}): {ph_train} vs {other_phs_train}')
                print('-'*100)                
                #########
                # TRAIN #
                #########
                models[classify_dim][i_split][train_at_feature][i_ph_train] = \
                    time_gen.fit(X_train, y_train)
            
                for test_at_feature in feature_names:
                    phones_test = phones[classify_dim][test_at_feature] # Take the corresponding phones in the test group (e.g., B, D, G)
                    ph_test = phones[classify_dim][test_at_feature][i_ph_train] # Take the corresponding phone in the test group based on index, i_ph_train (e.g., M -> B)
                    other_phs_test = [ph for ph in phones_test
                                      if ph !=  ph_test]
                    X_test, y_test, stimuli_test = \
                        lump_data_together(data_phones,
                                           ph_test, other_phs_test,
                                           i_split, 'test')
                    print(f'\nTest model at {test_at_feature} ({X_test.shape}, {y_test.shape}): {ph_test} vs {other_phs_test}')
                    
                    scores[classify_dim][i_split][train_at_feature][i_ph_train]\
                        [test_at_feature] = \
                            time_gen.score(X_test, y_test)
                    
############
# PLOTTING #
############
#for classify_dim in ['manner', 'place']: # 3-by-3 matrix of generalization GATs for each classify_dimension
    fig, axs = plt.subplots(3, 3, figsize=(15,15))
    for i_train_at_feature, train_at_feature in enumerate(feature_names):
        for i_test_at_feature, test_at_feature in enumerate(feature_names):
            GAT_scores = []
            for i_ph in range(3):
                for i_split in range(args.n_splits):
                    GAT_scores.append(scores[classify_dim][i_split][train_at_feature][i_ph][test_at_feature])
            mean_GAT_scores = np.mean(np.dstack(GAT_scores), axis = -1)
            # PLOT
            vmax = np.max(mean_GAT_scores)
            im = axs[2-i_train_at_feature, i_test_at_feature].matshow(mean_GAT_scores,
                                                              cmap='RdBu_r',
                                                              origin='lower',
                                                              extent=data.epochs[0].times[[0, -1, 0, -1]],
                                                              vmin=0.25,  # vmin=1-vmax,
                                                              vmax=0.75)  # vmax=vmax)
            axs[2-i_train_at_feature, i_test_at_feature].axhline(0., color='k')
            axs[2-i_train_at_feature, i_test_at_feature].axvline(0., color='k')
            axs[2-i_train_at_feature, i_test_at_feature].xaxis.set_ticks_position('bottom')
            axs[2-i_train_at_feature, i_test_at_feature].set_xticks(np.arange(0, data.epochs[0].tmax, 0.2))
            axs[2-i_train_at_feature, i_test_at_feature].set_yticks(np.arange(0, data.epochs[0].tmax, 0.2))
            axs[2-i_train_at_feature, i_test_at_feature].set_xlabel('Testing Time (s)')
            axs[2-i_train_at_feature, i_test_at_feature].set_title(f'{train_at_feature} -> {test_at_feature}')
            if i_test_at_feature == 0:
                axs[i_train_at_feature, i_test_at_feature].set_ylabel('Training Time (s)')
            else:
                axs[i_train_at_feature, i_test_at_feature].set_ylabel('')

            #ax.set_title(f'{args.comparison_name} {args.block_type} {args.comparison_name_test} {args.block_type_test}')
            plt.colorbar(im, ax=axs[i_train_at_feature, i_test_at_feature])


    ########
    # SAVE #
    ########

    if len(list(set(args.data_type))) == 1: args.data_type = list(set(args.data_type))
    if len(list(set(args.filter))) == 1: args.filter = list(set(args.filter))
    args.probe_name_lumpped = sorted(list(set([item for sublist in args.probe_name for item in sublist]))) # !! lump together all probe names !! to reduce filename length
    print(args.__dict__, list_args2fname)
    fname_fig = dict2filename(args.__dict__, '_', list_args2fname, 'png', True)
    fname_fig = os.path.join(args.path2figures, classify_dim + '_' + fname_fig)
    fig.savefig(fname_fig)
    print('Figures saved to: ' + fname_fig)
