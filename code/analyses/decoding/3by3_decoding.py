import argparse, os, sys
from functions import data_manip
from functions.classification import prepare_data_for_classifier
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
from sklearn.svm import LinearSVC
from mne.decoding import (cross_val_multiscore, LinearModel, GeneralizingEstimator)
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.multiclass import OneVsRestClassifier
parser = argparse.ArgumentParser(description='Generate plots for TIMIT experiment')
# DATA
parser.add_argument('--patient', nargs='*', default=[], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], action='append', default=[], help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='word', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'gaussian-kernel-25', 'high-gamma'], action='append', default=[], help='')
parser.add_argument('--probe-name', default=[], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
# QUERY
parser.add_argument('--classifier', default='ridge', choices=['svc', 'logistic', 'ridge'], help='Specify a classifier type to be used')
parser.add_argument('--min-trials', default=10, type=float, help='Minimum number of trials from each class.')
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
args.patient = ['patient_' + p for p in  args.patient]
args.query = ''
print(mne.__version__)

# Which args to have in fig filename
list_args2fname = ['patient', 'data_type', 'filter', 'level']
if args.responsive_channels_only: list_args2fname += ['responsive_channels_only']
if args.probe_name: list_args2fname.append('probe_name')
print(list_args2fname)

if not args.path2figures:
    args.path2figures = os.path.join('..', '..', 'Figures', 'Decoding')
print(args)

########
# DATA #
########
epochs_list = data_manip.load_neural_data(args)
if args.decimate:
    [epochs.decimate(args.decimate) for epochs in epochs_list]
print(epochs_list[0])
print('Channel names:')
[print(e.ch_names) for e in epochs_list]

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
n_splits = 5
time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=1)

# TRAIN
# RUNNING EXAMPLE FOR (M vs (N, NG)), tested on (B vs (D, G)):
models = {} # NESTED DICTS: models[dimension][train_feature][ph][CV-split]
scores = {} # NESTED DICTS: scores[dimension][train_feature][ph][CV-split][test_feature]
for dim in ['manner', 'place']: # classification along dimension (e.g., between place features)
    models[dim], scores[dim] = {}, {}
    feature_names = phones[dim].keys()
    for train_feature in feature_names: # fixed feature (e.g., nasals with different place features)
        models[dim][train_feature], scores[dim][train_feature] = {}, {}
        phones_train = phones[dim][train_feature]
        for i_ph_train, ph_train in enumerate(phones_train): # which phone in the triplet (e.g., /M/ vs (/N/ and /NG/)
            models[dim][train_feature][i_ph_train], scores[dim][train_feature][i_ph_train] = {}, {}
            # GET DATA FOR TRAIN FEATURE (e.g., nasals: M vs (N, NG))
            other_phs_train = [ph for ph in phones_train if ph !=  ph_train]
            query_1_train_feature = f'(block in [2, 4, 6]) and (phone_string=="{ph_train}")'
            query_2_train_feature = f'(block in [2, 4, 6]) and (phone_string in {other_phs_train})'
            queries_train_feature = [query_1_train_feature, query_2_train_feature]
            X_train_feature, y_train_feature, stimuli_train_feature = prepare_data_for_classifier(epochs_list, queries_train_feature)
            cv = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
            for i_split, (IXs_train_feature_train, IXs_train_feature_test) in enumerate(cv.split(X_train_feature, y_train_feature)):
                scores[dim][train_feature][i_ph_train][i_split] = {}
                # TRAIN ON CURRENT SPLIT
                print('\nTrain model for:')
                print(dim, train_feature, ph_train, i_split)
                print(X_train_feature.shape, y_train_feature.shape)
                print(X_train_feature[IXs_train_feature_train].shape, y_train_feature[IXs_train_feature_train].shape)
                models[dim][train_feature][i_ph_train][i_split] = time_gen.fit(X_train_feature[IXs_train_feature_train], y_train_feature[IXs_train_feature_train])
            
                for test_feature in feature_names:
                    # GET DATA FOR TEST FEATURE (e.g., voiced plosvies: B vs (D, G))
                    phones_test = phones[dim][test_feature] # Take the corresponding phones in the test group (e.g., B, D, G)
                    ph_test = phones[dim][test_feature][i_ph_train] # Take the corresponding phone in the test group based on index, i_ph_train (e.g., M -> B)
                    other_phs_test = [ph for ph in phones_test if ph !=  ph_test]
                    query_1_test_feature = f'(block in [2, 4, 6]) and (phone_string=="{ph_test}")'
                    query_2_test_feature = f'(block in [2, 4, 6]) and (phone_string in {other_phs_test})'
                    queries_test_feature = [query_1_test_feature, query_2_test_feature]
                    X_test_feature, y_test_feature, stimuli_test_feature = prepare_data_for_classifier(epochs_list, queries_test_feature)
                    
                    for i_split_test, (IXs_test_feature_train, IXs_test_feature_test) in enumerate(cv.split(X_test_feature, y_test_feature)):
                        if i_split == i_split_test:
                            # SCORE
                            print(f'{dim}: {train_feature} -> {test_feature}')
                            print(f'{dim}: Testing the model trained on {train_feature} (between {ph_train} and {other_phs_train}) on {test_feature} (between {ph_test} and {other_phs_test})')
                            print(f'Split: {i_split}')
                            print(queries_train_feature)
                            print(queries_test_feature)
                            print(X_test_feature[IXs_test_feature_test].shape, y_test_feature[IXs_test_feature_test].shape)
                            scores[dim][train_feature][i_ph_train][i_split][test_feature] = time_gen.score(X_test_feature[IXs_test_feature_test], y_test_feature[IXs_test_feature_test])
                    print('-' * 120)

############
# PLOTTING #
############
#for dim in ['manner', 'place']: # 3-by-3 matrix of generalization GATs for each dimension
    feature_names = phones[dim].keys()
    fig, axs = plt.subplots(3, 3, figsize=(15,15))
    for i_train_feature, train_feature in enumerate(feature_names):
        for i_test_feature, test_feature in enumerate(feature_names):
            # COLLECT ACROSS SPLITS AND ACROSS one-vs-all AND MEAN SCORES
            GAT_scores = []
            for i_ph in range(3):
                for i_split in range(n_splits):
                    GAT_scores.append(scores[dim][train_feature][i_ph][i_split][test_feature])
            mean_GAT_scores = np.mean(np.dstack(GAT_scores), axis = -1)
            #print(GAT_scores) 
            #print(mean_GAT_scores) 
            # PLOT
            vmax = np.max(mean_GAT_scores)
            im = axs[i_train_feature, i_test_feature].matshow(mean_GAT_scores, cmap='RdBu_r', origin='lower', extent=epochs_list[0].times[[0, -1, 0, -1]], vmin=1-vmax, vmax=vmax)
            axs[2-i_train_feature, i_test_feature].axhline(0., color='k')
            axs[2-i_train_feature, i_test_feature].axvline(0., color='k')
            axs[2-i_train_feature, i_test_feature].xaxis.set_ticks_position('bottom')
            axs[2-i_train_feature, i_test_feature].set_xticks(np.arange(0, epochs_list[0].tmax, 0.2))
            axs[2-i_train_feature, i_test_feature].set_yticks(np.arange(0, epochs_list[0].tmax, 0.2))
            axs[2-i_train_feature, i_test_feature].set_xlabel('Testing Time (s)')
            if i_test_feature == 0:
                axs[i_train_feature, i_test_feature].set_ylabel('Training Time (s)')
            else:
                axs[i_train_feature, i_test_feature].set_ylabel('')

            #ax.set_title(f'{args.comparison_name} {args.block_type} {args.comparison_name_test} {args.block_type_test}')
            plt.colorbar(im, ax=axs[i_train_feature, i_test_feature])


    ########
    # SAVE #
    ########

    if len(list(set(args.data_type))) == 1: args.data_type = list(set(args.data_type))
    if len(list(set(args.filter))) == 1: args.filter = list(set(args.filter))
    args.probe_name = sorted(list(set([item for sublist in args.probe_name for item in sublist]))) # !! lump together all probe names !! to reduce filename length
    print(args.__dict__, list_args2fname)
    fname_fig = dict2filename(args.__dict__, '_', list_args2fname, 'png', True)
    fname_fig = os.path.join(args.path2figures, dim + '_' + fname_fig)
    fig.savefig(fname_fig)
    print('Figures saved to: ' + fname_fig)
