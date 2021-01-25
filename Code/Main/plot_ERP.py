import argparse, os, glob, sys
# Set current working directory to that of script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
import mne
import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy import percentile
from sklearn import linear_model
from sklearn.metrics import r2_score
from operator import itemgetter
from pprint import pprint
from functions import comparisons_phone, comparisons_word, comparisons_sentence_onset, load_settings_params

parser = argparse.ArgumentParser(description='Generate plots')
parser.add_argument('--patient', default='479_11', help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], default='micro', help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='sentence_onset', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'high-gamma'], default='high-gamma', help='')
parser.add_argument('--probe-name', default=[], action='append', type=str, help='Probe name to plot (will ignore args.channel), e.g., LSTG')
parser.add_argument('--comparison-name', default=[], help='int. Comparison name from Code/Main/functions/comparisons_level.py. see print_comparisons.py')
parser.add_argument('--block-type', choices=['auditory', 'visual'], default='auditory', help='Block type will be added to the query in the comparison')
parser.add_argument('--fixed-constraint', default=[], help='A fixed constrained added to query. For example first_phone == 1 for auditory blocks')
parser.add_argument('--tmin', default=[], type=float, help='crop window. If empty, only crops 0.1s from both sides, due to edge effects.')
parser.add_argument('--tmax', default=[], type=float, help='crop window')
parser.add_argument('--channel', default=[], action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--baseline', default=[], type=str, help='Baseline to apply as in mne: (a, b), (None, b), (a, None), (None, None) or None')
parser.add_argument('--baseline-mode', choices=['mean', 'ratio', 'logratio', 'percent', 'zscore', 'zlogratio'], default='zscore', help='Type of baseline method')
parser.add_argument('--SOA', default=500, help='SOA in design [msec]')
parser.add_argument('--word-ON-duration', default=250, help='Duration for which word word presented in the RSVP [msec]')
parser.add_argument('--remove-outliers', action="store_true", default=False, help='Remove outliers based on percentile 25 and 75')
parser.add_argument('--dont-write', default=False, action='store_true', help="If True then file will be overwritten")
parser.add_argument('--verbose', default=False, action='store_true')

args = parser.parse_args()
args.patient = 'patient_' + args.patient
if isinstance(args.baseline, str):
    args.baseline = eval(args.baseline)
pprint(args)

print('Loading settings...')
settings = load_settings_params.Settings(args.patient)

# LOAD
fname = '%s_%s_%s_%s-epo.fif' % (args.patient, args.data_type, args.filter, args.level)
epochs = mne.read_epochs(os.path.join(settings.path2epoch_data, fname), preload=True)

# COMPARISON
if args.level == 'phone':
    comparisons = comparisons_phone.comparison_list()
elif args.level == 'word':
    comparisons = comparisons_word.comparison_list()
elif args.level == 'sentence_onset':
    comparisons = comparisons_sentence_onset.comparison_list()
elif args.level == 'sentence_offset':
    comparisons = comparisons_sentence_offset.comparison_list()
else:
    raise('Wrong level argument')
comparison = comparisons[args.comparison_name]

block_str=' and (block in [1, 3, 5])' if args.block_type == 'visual' else ' and (block in [2, 4, 6])'
queries_to_compare = []
if isinstance(comparison['queries'], str): # if string (instead of a list of strings) then queries are based on all values.
    all_possible_values = list(set(epochs.metadata[comparison['queries']]))
    for val in all_possible_values:
        query = comparison['queries'] + ' == ' + str(val)
        query += block_str
        if args.fixed_constraint:
            query += args.fixed_constraint
        condition_name = comparison['queries'] + ' == ' + str(val)
        color = np.random.rand(3,)
        queries_to_compare.append((condition_name, query, color))
else: # list of strings of queries
    if not comparison['colors']:
        for _ in comparison['queries']:
            comparison['colors'].append(np.random.rand(3,))
    for condition_name, query, color in zip(comparison['condition_names'], comparison['queries'], comparison['colors']):
        query += block_str
        if args.fixed_constraint:
            query += args.fixed_constraint
        queries_to_compare.append((condition_name, query, color))
print(args.comparison_name, args.block_type)
for query in queries_to_compare: print(query)

# PICK
if args.probe_name:
    picks = []
    for IX, ch_name in enumerate(epochs.ch_names):
        #if [i for i in ch_name if not i.isdigit()] == args.probe_name:
        for probe_name in args.probe_name:
            if probe_name in ch_name:
                picks.append(IX)
    epochs.pick(picks)
else:
    if args.channel:
        epochs.pick([int(c) for c in args.channel])

# BASELINE
if args.filter != 'high-gamma': # high-gamma is already baselined during epoching (generate_epochs.py)
    if args.baseline:
        print('Apply baseline:')
        epochs.apply_baseline(args.baseline, verbose=True)

# CROP
if args.tmin and args.tmax:
    epochs.crop(args.tmin, args.tmax)
else:
    if args.filter == 'high-gamma': # remove boundary effects
        epochs.crop(min(epochs.times) + 0.1, max(epochs.times) - 0.1)


for ch_name in epochs.info['ch_names']:
    print(ch_name)
    str_comparison = '_'.join([tup[0] for tup in queries_to_compare])
    fname_fig = 'ERP_evoked_%s_%s_%s_%s_%s_%s.png' % (args.patient, args.data_type, args.level, args.filter, ch_name, args.comparison_name)
    fname_fig = os.path.join(settings.path2figures, fname_fig) 
    if (not os.path.exists(fname_fig)) or (not args.dont_write): # Check if output fig file already exists: 
        evoked_dict = dict()
        colors_dict = {}
        for i, (condition_name, query, color) in enumerate(queries_to_compare):
            if args.verbose:
                print(list(epochs[query].metadata[args.level.split('_')[0] + '_string']))
                print(list(epochs[query].metadata['sentence_string']))
            colors_dict[condition_name] = color 
            evoked_dict[condition_name] = epochs[query].average(method='median')
        fig = mne.viz.plot_compare_evokeds(evoked_dict, show=False, colors=colors_dict, picks=ch_name)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        plt.subplots_adjust(right=0.6)
        
        # SAVE:
        plt.savefig(fname_fig)
        print('fig saved to: ' + fname_fig)
        plt.close('All')



#if not os.path.exists(args.path2figures):
#    os.makedirs(args.path2figures)
