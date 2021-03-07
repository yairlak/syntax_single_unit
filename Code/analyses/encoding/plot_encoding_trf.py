#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:49:54 2021

@author: yl254115
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:48:59 2021

@author: yl254115
"""

import argparse, os, sys, pickle
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
from utils.utils import dict2filename
from utils.data_manip import load_neural_data
from utils.features import get_features
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Train an encoding model on neural data')
# DATA
parser.add_argument('--patient', action='append', default=['479_11'], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], action='append', default=['micro'], help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='word', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'gaussian-kernel-25', 'high-gamma'], action='append', default=['gaussian-kernel'], help='')
parser.add_argument('--probe-name', default=[['LSTG']], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channe-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
# MISC
parser.add_argument('--path2output', default=os.path.join('..', '..', '..', 'Output', 'encoding_models'), help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--path2figures', default=os.path.join('..', '..', '..', 'Figures', 'encoding_models'), help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--decimate', default=[], type=float, help='If not empty, (for speed) decimate data by the provided factor.')
parser.add_argument('--model-type', default='ridge', choices=['ridge', 'lasso', 'standard']) 
parser.add_argument('--query', default=['block in [2]'], help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')

#############
# USER ARGS #
#############
args = parser.parse_args()
args.patient = ['patient_' + p for p in  args.patient]
args.block_type = 'both'
print('args\n', args)
assert len(args.patient)==len(args.data_type)==len(args.filter)==len(args.probe_name)
# FNAME 
#list_args2fname = ['patient', 'data_type', 'filter', 'level', 'block_type', 'model_type', 'ch_name', 'feature_list', 'query']
list_args2fname = ['patient', 'data_type', 'filter', 'level', 'model_type', 'query', 'probe_name']

if not os.path.exists(args.path2figures):
    os.makedirs(args.path2figures)

#########################
# LOAD ENCODING RESULTS #
#########################
# epochs_list = load_neural_data(args)
# args.ch_name = epochs_list[0].ch_names[0]
# args2fname = args.__dict__.copy()
# fname = dict2filename(args2fname, '_', list_args2fname, '', True)
# os.path.join(args.path2output, fname)
# with open(os.path.join(args.path2output, fname + '.pkl'), 'rb') as f:
#     model, scores, args_encoding = pickle.load(f)

# args.tmin, args.tmax = args_encoding.tmin, args_encoding.tmax
# args.decimate = args_encoding.decimate

# epochs_list = load_neural_data(args)

# metadata = epochs_list[0].metadata
# times = epochs_list[0].times

# _, feature_values, feature_info, feature_groups = get_features(metadata, []) # GET DESIGN MATRIX


# for epochs in epochs_list:
   
    # ADD CH_NAME TO FNAME
args2fname = args.__dict__.copy()
fname = dict2filename(args2fname, '_', list_args2fname, '', True)


os.path.join(args.path2output, fname)

rfs, scores, args_trf, feature_info_all = pickle.load(open(os.path.join(args.path2output, fname + '.pkl'), 'rb'))
feature_info_all['phonological_features']['color'] = 'm'

feature_names = list(scores.keys())
feature_names.remove('full')

num_features = len(feature_names)

############
# PLOTTING #
############
rf = rfs[0] # FULL MODEL
for i_channel in range(8):
    times = rf.delays_*1000/rf.sfreq # in [msec]
    coef = rf.coef_[i_channel, :, :]
    fig, ax = plt.subplots(figsize=(20,10))
    for i_feature, feature_name in enumerate(feature_names):
        scores_full = scores['full'][0][i_channel]
        scores_reduced = scores[feature_name][0][i_channel]
        # coef of current feature
        
        IXs = feature_info_all[feature_name]['IXs']
        coef_curr_feature = np.max(coef[IXs[0]:IXs[1], :], axis=0)
        
        if feature_info_all[feature_name]['color']:
            color = feature_info_all[feature_name]['color']
        else:
            color = None
        if ('ls' in feature_info_all[feature_name].keys()) and feature_info_all[feature_name]['ls']:
            ls = feature_info_all[feature_name]['ls']
        else:
            ls = '-'
        if ('lw' in feature_info_all[feature_name].keys()) and feature_info_all[feature_name]['lw']:
            lw = feature_info_all[feature_name]['lw']
        else:
            lw = 3
        
        ax.plot(times, coef_curr_feature, color=color, ls=ls, lw=lw, label=feature_name)
        print(feature_name, color)
        # ax.fill_between(times*1e3, effect_size_mean + effect_size_std, effect_size_mean - effect_size_std , color=color, alpha=0.2)
    
    ax.legend(loc='center left', bbox_to_anchor=(1.12, 0.5), ncol=int(np.ceil(num_features/40)))
    ax.set_xlabel('Time (msec)', fontsize=20)
    ax.set_ylabel(r'Beta', fontsize=20)
    ax.set_ylim((None, None)) 
    if args.block_type == 'visual':
        ax.axvline(x=0, ls='--', color='k')
        ax.axvline(x=500, ls='--', color='k')
    ax.axhline(ls='--', color='k')
    ax.set_title(f'corrcoef = {scores_full:1.2f}')

    # color = 'k'
    # ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_ylabel('Coefficient of determination ($R^2$)', color=color, fontsize=20)  # we already handled the x-label with ax1
    # ax2.plot(times*1e3, r2_full_mean, color=color, lw=3)
    # ax2.fill_between(times*1e3, r2_full_mean+r2_full_std, r2_full_mean-r2_full_std, color=color, alpha=0.2)
    # ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_ylim((0, 1)) 
    plt.subplots_adjust(right=0.8)
    
    fname_fig = os.path.join(args.path2figures, fname + f'_{i_channel+1}_groupped.png')
    fig.savefig(fname_fig)
    plt.close(fig)
    print('Figure saved to: ', fname_fig)
    
