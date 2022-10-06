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
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from decoding.decoder import decode_comparison

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
                    default='auditory',
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
parser.add_argument('--equalize-classes', default='downsample',
                    choices=['upsample', 'downsample'])
parser.add_argument('--gat', default=False, action='store_true',
                    help='If True, GAT will be computed; else, diagonal only')
# MISC
parser.add_argument('--coords', default=None, type=float, nargs='*',
                    help="coordinates (e.g., MNI) for searchlight")
parser.add_argument('--side', default=8, type=float, help='Side of cube in mm')
parser.add_argument('--tmin', default=None, type=float)
parser.add_argument('--tmax', default=None, type=float)
#parser.add_argument('--vmin', default=None, type=float, help='')
#parser.add_argument('--vmax', default=None, type=float, help='')
parser.add_argument('--decimate', default=50, type=int)
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

# CONTRAST
print('\nCONTRASTS:')
metadata = data.epochs[0].metadata
comparisons[0] = update_queries(comparisons[0], args.block_train, # TRAIN
                                args.fixed_constraint, metadata)
comparisons[1] = update_queries(comparisons[1], args.block_test, # TEST
                                args.fixed_constraint_test, metadata)
[pprint(comparison) for comparison in comparisons] 

# DECODE
scores, pvals, temp_estimator, clf, stimuli, stimuli_gen = \
                    decode_comparison(data.epochs, comparisons, args)

# SAVE
args2fname = get_args2fname(args)  # List of args
fname_pkl = dict2filename(args.__dict__, '_', args2fname, 'pkl', True)
fname_pkl = os.path.join(args.path2output, fname_pkl)
with open(fname_pkl, 'wb') as f:
    pickle.dump([scores, pvals, data.epochs[0].times,
                 temp_estimator, clf, comparisons,
                 (stimuli, stimuli_gen), args], f)
print(f'Results saved to: {fname_pkl}')
