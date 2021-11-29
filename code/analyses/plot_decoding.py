
import argparse, os, sys, pickle
import mne
from utils.data_manip import DataHandler
from utils import classification, comparisons, load_settings_params, data_manip
from utils.utils import dict2filename, update_queries, probename2picks
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from mne.decoding import (cross_val_multiscore, LinearModel, GeneralizingEstimator, SlidingEstimator)
import numpy as np
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.multiclass import OneVsRestClassifier
from utils.utils import get_patient_probes_of_region, get_all_patient_numbers
import uuid
import copy
from decoding.utils import get_args2fname, update_args, get_comparisons 

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(description='')
# DATA
parser.add_argument('--patient', action='append', default=[], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], action='append', default=[], help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default=None, help='')
parser.add_argument('--filter', choices=['raw', 'high-gamma'], action='append', default=[], help='')
parser.add_argument('--probe-name', default=[], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
parser.add_argument('--ROIs', default=None, nargs='*', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--data-type_filters', choices=['micro_high-gamma','macro_high-gamma', 'micro_raw','macro_raw', 'spike_raw'], nargs='*', default=[], help='Only if args.ROIs is used')
parser.add_argument('--smooth', default=None, type=int, help='If not empty, (for speed) decimate data by the provided factor.')
# QUERY
parser.add_argument('--comparison-name', default=None, help='Comparison name from Code/Main/functions/comparisons.py')
parser.add_argument('--comparison-name-test', default=None, help='Comparison name from Code/Main/functions/comparisons.py')
parser.add_argument('--block-train', choices=['auditory', 'visual'], default='visual', help='Block type will be added to the query in the comparison')
parser.add_argument('--block-test', choices=['auditory', 'visual'], default=None, help='Block type will be added to the query in the comparison')
parser.add_argument('--fixed-constraint', default=None, help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--fixed-constraint-test', default=None, help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--min-trials', default=10, type=float, help='Minimum number of trials from each class.')
# DECODER
parser.add_argument('--classifier', default='ridge', choices=['svc', 'logistic', 'ridge'], help='Specify a classifier type to be used')
parser.add_argument('--gat', default=False, action='store_true', help='If True, GAT will be computed; else, diagonal only')
# MISC
parser.add_argument('--tmin', default=None, type=float, help='crop window. If empty, only crops 0.1s from both sides, due to edge effects.')
parser.add_argument('--tmax', default=None, type=float, help='crop window')
parser.add_argument('--vmin', default=None, type=float, help='')
parser.add_argument('--vmax', default=None, type=float, help='')
parser.add_argument('--decimate', default=None, type=int, help='If not empty, (for speed) decimate data by the provided factor.')
parser.add_argument('--cat-k-timepoints', type=int, default=1, help='How many time points to concatenate before classification')
parser.add_argument('--path2figures', default='../../Figures/Decoding')
parser.add_argument('--path2output', default='../../Output/decoding')
parser.add_argument('--dont-overwrite', default=False, action='store_true', help="If True then file will be overwritten")

# PARSE
args = parser.parse_args()
args = update_args(args)
args2fname = get_args2fname(args) # List of args

###############
# LOAD SCORES #
###############
fname_pkl = dict2filename(args.__dict__, '_', args2fname, 'pkl', True)
fname_pkl = os.path.join(args.path2output, fname_pkl)
results = pickle.load(open(fname_pkl, 'rb'))
scores, pvals, times, time_gen, clf, comparisons, stimuli, args_decoding = results
tmin, tmax = times.min(), times.max()
############
# PLOTTING #
############
fig, ax = plt.subplots(1, figsize=(10,10))
vmax = np.max(np.mean(scores, axis=0))
chance_level = 1/len(comparisons[0]['queries'])
if args_decoding.gat:
    im = ax.matshow(np.mean(scores, axis=0), cmap='RdBu_r', origin='lower', extent=times[[0, -1, 0, -1]], vmin=1-vmax, vmax=vmax)
    ax.axhline(0., color='k')
    ax.axvline(0., color='k')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(tmin, tmax, 0.2))
    ax.set_yticks(np.arange(tmin, tmax, 0.2))
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title(f'{args.comparison_name} {args.block_train} {args.comparison_name_test} {args.block_test}')
    plt.colorbar(im, ax=ax)
else:
    ax.plot(times, np.mean(scores, 0),label='score', color='k', lw=3)
    y_min = np.mean(scores, 0) - np.std(scores, 0)/np.sqrt(scores.shape[0])
    y_max = np.mean(scores, 0) + np.std(scores, 0)/np.sqrt(scores.shape[0])
    ax.fill_between(times, y_min, y_max, alpha=0.2, color='g')
    ax.axhline(chance_level, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Time', fontsize=30)
    ax.set_ylabel('AUC', fontsize=30)  # Area Under the Curve
    ax.legend(fontsize=30)
    ax.axvline(.0, color='k', linestyle='-')
    my_xticks = ax.get_xticks()
    my_yticks = ax.get_yticks()
    plt.xticks(np.arange(tmin, tmax, 0.2), visible=True, fontsize=30)
    plt.yticks([chance_level, my_yticks[-1], 1], visible=True, fontsize=30)
    if args_decoding.classifier not in ['ridge']:
        plt.ylim([0, 1])
    #ax.set_title(f'{args.comparison_name} {args.block_train} {args.comparison_name_test} {args.block_test}')

########
# SAVE #
########
#if len(list(set(args.data_type))) == 1: args.data_type = list(set(args.data_type))
#if len(list(set(args.filter))) == 1: args.filter = list(set(args.filter))
#args.probe_name = sorted(list(set([item for sublist in args.probe_name for item in sublist]))) # !! lump together all probe names !! to reduce filename length
##args.data_type = sorted(list(set([item for sublist in args.data_type for item in sublist]))) # !! lump together all probe names !! to reduce filename lengt
##args.filter = sorted(list(set([item for sublist in args.filter for item in sublist]))) # !! lump together all probe names !! to reduce filename lengt
#args.patient = [p[6:] for p in args.patient]

if args_decoding.gat:
    init = 'GAT_'
else:
    init = 'Slider_'
#print(args.__dict__, list_args2fname)
fname_fig = dict2filename(args.__dict__, '_', args2fname, 'png', True)
fname_fig = os.path.join(args.path2figures, init + fname_fig)
if len(fname_fig)>4096:
    fname_fig = str(uuid.uuid4())    
fig.savefig(fname_fig)
print('Figures saved to: ' + fname_fig)
