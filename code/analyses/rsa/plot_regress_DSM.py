import argparse, os, sys, pickle
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
import csv
import mne
from functions import classification, comparisons, load_settings_params
from functions.utils import dict2filename, update_queries, probename2picks, pick_responsive_channels
from functions.data_manip import load_epochs_data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

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
parser.add_argument('--model-type', default='euclidean', choices=['euclidean', 'logistic', 'lstm', 'cnn']) # 'svc' and 'ridge' are omited since they don't implemnent predict_proba (although there's a work around, using their decision function and map is to probs with eg softmax)
# QUERY
parser.add_argument('--comparison-name', default='all_words', help='Comparison name from Code/Main/functions/comparisons.py')
parser.add_argument('--comparison-name-test', default=[], help='Comparison name from Code/Main/functions/comparisons.py')
parser.add_argument('--block-type', choices=['auditory', 'visual'], default='auditory', help='Block type will be added to the query in the comparison')
parser.add_argument('--block-type-test', choices=['auditory', 'visual', []], default=[], help='Block type will be added to the query in the comparison')
parser.add_argument('--fixed-constraint', default=[], help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--fixed-constraint-test', default=[], help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--classifier', default='logistic', choices=['svc', 'logistic', 'ridge'], help='Specify a classifier type to be used')
parser.add_argument('--label-from-metadata', default=[], help='Field name in metadata that will be used to generate labels for the different classes. If empty, condition_names in comparison will be used')
# FEATURES
parser.add_argument('--path2features', default=None, type=str, help='')
parser.add_argument('--class-names', default=[], type=str, nargs='*', help='')
# MISC
parser.add_argument('--vmin', default=None, type=float, help='')
parser.add_argument('--vmax', default=None, type=float, help='')
parser.add_argument('--times', nargs='*', default=[0.1], type=float, help='')
parser.add_argument('--num-bins', default=1, type=int, help='')
parser.add_argument('--time-window', default=0.1, type=float, help='')
parser.add_argument('--min-trials', default=10, type=float, help='Minimum number of trials from each class.')
parser.add_argument('--decimate', default=[], type=float, help='If not empty, (for speed) decimate data by the provided factor.')
parser.add_argument('--cat-k-timepoints', type=int, default=1, help='How many time points to concatenate before classification')
parser.add_argument('--path2figures', default=[], help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--path2output', default=[], help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--dont-overwrite', default=False, action='store_true', help="If True then file will be overwritten")
# PARSE
args = parser.parse_args()
#args.patient = ['patient_' + p for p in  args.patient]
print(mne.__version__)

# Which args to have in fig filename
#list_args2fname = ['patient', 'data_type', 'filter', 'level', 'comparison_name', 'block_type']
list_args2fname = ['patient', 'data_type', 'filter', 'level', 'comparison_name', 'block_type', 'time_window', 'num_bins', 'min_trials']
if args.block_type_test: list_args2fname += ['comparison_name_test', 'block_type_test']
if args.probe_name: list_args2fname.append('probe_name')
if args.responsive_channels_only: list_args2fname += ['responsive_channels_only']
print(list_args2fname)

if not args.path2figures:
    args.path2figures = os.path.join('..', '..', 'Figures', 'RSA')
if not args.path2output:
    args.path2output = os.path.join('..', '..', 'Output', 'RSA')
print(args)

times, coefs_over_time, scores_over_time = [], [], []
for t in args.times:
    try:
        args2fname = args.__dict__.copy()
        if len(list(set(args2fname['data_type']))) == 1: args2fname['data_type'] = list(set(args2fname['data_type']))
        if len(list(set(args2fname['filter']))) == 1: args2fname['filter'] = list(set(args2fname['filter']))
        args2fname['probe_name'] = sorted(list(set([item for sublist in args2fname['probe_name'] for item in sublist]))) # !! lump together all probe names !! to reduce filename length
        if 'time' not in list_args2fname: list_args2fname.append('time')
        args2fname['time'] = t

        fname_regress = dict2filename(args2fname, '_', list_args2fname, 'pkl', True)
        fname_regress = os.path.join(args.path2output, 'RegCoef_' + args.model_type + '_' + fname_regress)
        print('Loading: ', fname_regress)
        with open(fname_regress, 'rb') as f:
            models, results, args_conf, class_names_features = pickle.load(f)
        print(t)
        times.append(t)
        coefs = np.asarray([model.coef_ for model in models])
        coefs_over_time.append(coefs) # coefs: ndarray num_cv_folds X num_features
        # SCORES
        scores = [results[k]['scores'] for k in results.keys()]
        scores_over_time.append(scores)
    except:
        print('Failed on time point: ', t)
coefs_over_time = np.asarray(coefs_over_time) # num_features X num_cv_folds X num_timepoints
coefs_over_time = coefs_over_time.swapaxes(0, 2) # num_features X num_cv_folds X num_timepoints
#print(coefs_over_time)
scores_over_time = np.asarray(scores_over_time).transpose()
#print(scores_over_time, scores_over_time.shape)

class_names_features = ['Unigrams', 'Bigrams', 'Trigrams']
fig, ax = plt.subplots(figsize=(10, 10))
for f, coefs_curr_feature in enumerate(coefs_over_time):
    #print(coefs_curr_feature)
    #coefs_curr_feaure = np.sqrt(coefs_curr_feature) # W coefs is a PSD, the weights for the regression are the sqrt of its diagonal 
    coefs_mean = np.mean(coefs_curr_feature, axis=0)
    coefs_std = np.std(coefs_curr_feature, axis=0)
    ax.plot(times, coefs_mean, lw=3, label=class_names_features[f])
    #print(coefs_std)
    plt.fill_between(times, coefs_mean + coefs_std, coefs_mean - coefs_std, alpha=0.2)

ax.set_xticks(times)
ax.set_xticklabels(times)
ax.set_xlabel('Regression Time', fontsize=16)
ax.set_ylabel('Weight size', fontsize=16)
#ax.set_ylim([0, 0.4])
plt.legend()
#plt.title(set(args.data_type), fontsize=16)

fname_fig = dict2filename(args2fname, '_', list_args2fname, 'png', True)
fname_fig = os.path.join(args.path2figures, 'RegCoef_' + fname_fig)
plt.savefig(fname_fig)
print('Saved to:', fname_fig)
plt.close(fig)
##########
# SCORES #
##########
fig, ax = plt.subplots(figsize=(10, 10))
#for f, scores in enumerate(scores_over_time):
    #print(coefs_curr_feature)
    #coefs_curr_feaure = np.sqrt(coefs_curr_feature) # W coefs is a PSD, the weights for the regression are the sqrt of its diagonal 
scores_mean = np.mean(scores_over_time, axis=0)
scores_std = np.std(scores_over_time, axis=0)
#print(scores, scores_mean, scores_std)
ax.plot(times, scores_mean, lw=3)
#print(coefs_std)
plt.fill_between(times, scores_mean + scores_std, scores_mean - scores_std, alpha=0.2)

ax.set_xticks(times)
ax.set_xticklabels(times)
ax.set_xticks(times[::5])
ax.set_xlabel('Regression Time', fontsize=16)
ax.set_ylabel('Coefficient of determination ($R^2$)', fontsize=16)
ax.set_ylim([-0.2, 1])
#plt.legend()
#plt.title(set(args.data_type), fontsize=16)

fname_fig = dict2filename(args2fname, '_', list_args2fname, 'png', True)
fname_fig = os.path.join(args.path2figures, 'RegScore_' + fname_fig)
plt.savefig(fname_fig)
print('Saved to:', fname_fig)
