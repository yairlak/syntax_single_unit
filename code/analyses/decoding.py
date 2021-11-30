# -*- coding: utf-8 -*-

import argparse, os, pickle
import numpy as np
from scipy import stats
from decoding.utils import get_args2fname, update_args, get_comparisons
from decoding.data_manip import prepare_data_for_classification
from decoding.models import define_model
from decoding.data_manip import get_data
from sklearn.model_selection import LeaveOneOut, KFold
from utils.utils import dict2filename
from utils.utils import update_queries
from pprint import pprint
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(description='')
# DATA
parser.add_argument('--patient', action='append', default=[],
                    help='Patient number')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'],
                    action='append', default=[])
parser.add_argument('--level',
                    choices=['sentence_onset','sentence_offset',
                             'word', 'phone'],
                    default=None)
parser.add_argument('--filter', choices=['raw', 'high-gamma'],
                    action='append', default=[], help='')
parser.add_argument('--probe-name', default=[], nargs='*', action='append',
                    type=str, help='e.g., LSTG, overrides channel_name/num')
parser.add_argument('--ROIs', default=None, nargs='*', type=str,
                    help='e.g., Brodmann.22-lh, overrides probe_name')
parser.add_argument('--channel-name', default=[], nargs='*', action='append',
                    type=str, help='e.g., GA1-LAH1')
parser.add_argument('--channel-num', default=[], nargs='*', action='append',
                    type=int, help='e.g., 3 to pick the third channel')
parser.add_argument('--responsive-channels-only', action='store_true',
                    default=False, help='Based on aud and vis files in Epochs folder')
parser.add_argument('--data-type_filters',
                    choices=['micro_high-gamma','macro_high-gamma',
                             'micro_raw','macro_raw', 'spike_raw'], nargs='*',
                             default=[], help='Only if args.ROIs is used')
parser.add_argument('--smooth', default=None, type=int,
                    help='gaussian width in [msec]')
# QUERY
parser.add_argument('--comparison-name', default=None,
                    help='See Utils/comparisons.py')
parser.add_argument('--comparison-name-test', default=None,
                    help='See Utils/comparisons.py')
parser.add_argument('--block-train', choices=['auditory', 'visual'],
                    default=None,
                    help='Block type is added to the query in the comparison')
parser.add_argument('--block-test', choices=['auditory', 'visual'],
                    default=None,
                    help='Block type is added to the query in the comparison')
parser.add_argument('--fixed-constraint', default=None,
                    help='e.g., "and first_phone == 1"')
parser.add_argument('--fixed-constraint-test', default=None,
                    help='e.g., "and first_phone == 1"')
parser.add_argument('--min-trials', default=10, type=float,
                    help='Minimum number of trials from each class.')
# DECODER
parser.add_argument('--classifier', default='logistic',
                    choices=['svc', 'logistic', 'ridge'])
parser.add_argument('--gat', default=False, action='store_true',
                    help='If True, GAT will be computed; else, diagonal only')
# MISC
parser.add_argument('--tmin', default=None, type=float)
parser.add_argument('--tmax', default=None, type=float)
#parser.add_argument('--vmin', default=None, type=float, help='')
#parser.add_argument('--vmax', default=None, type=float, help='')
parser.add_argument('--decimate', default=None, type=int)
parser.add_argument('--cat-k-timepoints', type=int, default=1,
                    help='How many time points to concatenate before classification')
parser.add_argument('--path2figures', default='../../Figures/Decoding')
parser.add_argument('--path2output', default='../../Output/decoding')
parser.add_argument('--dont-overwrite', default=False, action='store_true',
                    help="If True then will not regenerate already existing figures")

args = parser.parse_args()
# CHECK AND UPDATE USER ARGUMENTS
args = update_args(args)

# GET COMPARISONS (CONTRASTS)
comparisons = get_comparisons(args.comparison_name, # List with two dicts for
                              args.comparison_name_test) # comparison train and test

print('\nARGUMENTS:')
pprint(args.__dict__, width=1)
if 'level' in comparisons[0].keys():
    args.level = comparisons[0]['level']
if len(comparisons[0]['queries'])>2:
    args.multi_class = True
else:
    args.multi_class = False

# LOAD DATA
print('\nLOADING DATA:')
data = get_data(args)

print('\nCONTRASTS:')
metadata = data.epochs[0].metadata
comparisons[0] = update_queries(comparisons[0], args.block_train, 
                                args.fixed_constraint, metadata)
comparisons[1] = update_queries(comparisons[1], args.block_test,
                                args.fixed_constraint_test, metadata)
[pprint(comparison) for comparison in comparisons] 


# PREPARE DATA FOR DECODING
print('\nPERPARING DATA FOR CLASSIFICATION:')
X, y, stimuli= prepare_data_for_classification(data.epochs,
                                               comparisons[0]['queries'],
                                               args.classifier,
                                               args.min_trials,
                                               verbose=True)
stimuli_gen = []
if args.GAC or args.GAM:
    if args.GAC: print('-'*30, '\nGeneralization Across Conditions\n', '-'*30)
    if args.GAM: print('-'*30, '\nGeneralization Across Modalities\n', '-'*30)
    X_gen, y_gen, stimuli_gen= prepare_data_for_classification(data.epochs,
                                                               comparisons[1]['queries'],
                                                               args.min_trials,
                                                               args.classifier,
                                                               verbose=True)

   
# SET A MODEL (CLASSIFIER)
clf, time_gen = define_model(args)


# LEAVE-ONE-OUT EVALUATION 
print('\n', '-'*40, 
      f'\nTraining a {args.classifier} model for a {len(list(set(y)))}-class problem\n', '-'*40)
loo = KFold(X.shape[0], shuffle=True, random_state=1)
scores = []
for i_split, (IXs_train, IX_test) in enumerate(loo.split(X, y)):
    # TRAIN MODEL
    if (args.GAC or args.GAM) and i_split == 0: # Use all training data once
        time_gen.fit(X, y)
    else:
        time_gen.fit(X[IXs_train], y[IXs_train])

    # EVAL MODEL
    if (args.GAC or args.GAM): # Use all training data once
        y_hat = time_gen.predict(X_gen[IX_test])
        if args.classifier in ['ridge']:
            score = np.power(y_hat - y_gen[IX_test], 2) # Squared error
        else:
            score = np.rint(y_hat) == y_gen[IX_test] # append a boolean vector,
        scores.append(score) # Store
        # indicating at which timepoints the model successfully predicted the class
    else:
        y_hat = time_gen.predict(X[IX_test])
        if args.classifier in ['ridge']:
            score = np.power(y_hat - y[IX_test], 2) # Squared error
        else:
            score = np.rint(y_hat) == y[IX_test] # append a boolean vector,
        scores.append(score) # Store

# The shape of scores is: num_splits X num_timepoints ( X num_timepoints)
scores = np.asarray(scores).squeeze()
print(f'Max score: {scores.mean(axis=0).max()}')

# STATS
if args.multi_class:
    pvals = []
else:
    pvals = np.asarray([stats.binom_test(success.sum(),
                                         n=len(success),
                                         p=0.5,
                                         alternative='greater') for success in scores.T])
# SAVE
args2fname = get_args2fname(args) # List of args
fname_pkl = dict2filename(args.__dict__, '_', args2fname, 'pkl', True)
fname_pkl = os.path.join(args.path2output, fname_pkl)
with open(fname_pkl, 'wb') as f:
     pickle.dump([scores, pvals, data.epochs[0].times, time_gen, clf, comparisons, (stimuli, stimuli_gen), args], f)
print(f'Results saved to: {fname_pkl}')
