#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:48:59 2021

@author: yl254115
"""
import argparse
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import sys
import pickle
from viz import plot_rf_coefs, plot_rf_r2
sys.path.append('..')
from utils.utils import dict2filename
import matplotlib.pyplot as plt
from encoding.models import TimeDelayingRidgeCV
import numpy as np
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser(description='Plot TRF results')
# DATA
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike'],
                    action='append', default=[], help='electrode type')
parser.add_argument('--filter', action='append',
                    default=[],
                    help='raw/high-gamma')
parser.add_argument('--smooth', default=25,
                    help='Gaussian smoothing in msec')
parser.add_argument('--probe-name', default=None, nargs='*',
                    action='append', type=str,
                    help='Probe name to plot (ignores channel-name/num)')
parser.add_argument('--channel-name', default=[], nargs='*', action='append',
                    type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=[], nargs='*', action='append',
                    type=int, help='channel number (if empty all channels)')
# MISC
parser.add_argument('--path2output',
                    default=os.path.join('..', '..', '..',
                                         'Output', 'encoding_models'))
parser.add_argument('--path2figures',
                    default=os.path.join('..', '..', '..',
                                         'Figures', 'encoding_models', 'scatters'))
parser.add_argument('--decimate', default=None, type=float,
                    help='If not empty, decimate data for speed.')
parser.add_argument('--model-type', default='ridge',
                    choices=['ridge', 'lasso', 'ridge_laplacian', 'standard'])
parser.add_argument('--ablation-method', default='remove',
                    choices=['shuffle', 'remove', 'zero'],
                    help='Method used to calcuated feature importance\
                        by reducing/ablating a feature family')
#parser.add_argument('--query-train', default="block in [2,4,6] and word_length>1")
#parser.add_argument('--query-test', default="block in [2,4,6] and word_length>1")
parser.add_argument('--each-feature-value', default=False, action='store_true',
                    help="Evaluate model after ablating each feature value. \
                         If false, ablate all feature values together")


#############
# USER ARGS #
#############

args = parser.parse_args()


def get_probe_name(channel_name, data_type):
    if data_type == 'micro':
        probe_name = channel_name[4:-1]
    elif data_type == 'macro':
        probe_name = channel_name[1:]
    elif data_type == 'spike':
        probe_name = channel_name
    else:
        print(channel_name, data_type)
        raise('Wrong data type')
    return probe_name

#columns = ['probe_name', 'hemisphere', 'ch_name', 'patient', 'feature', 'r']
df = pd.DataFrame()

print('Collecting results...')
patients="479_11 479_25 482 489 493 499 502 504 505 510 513 515 530 538 539"
dict_scatter = {}
for patient in patients.split():
    for block in ['auditory', 'visual']:

        args.query_train = {'auditory':'block in [2,4,6] and word_length>1',
                            'visual':'block in [1,3,5] and word_length>1'}[block]
        args.patient = ['patient_' + patient]
        list_args2fname = ['patient', 'data_type', 'filter', 'smooth',
                           'model_type', 'probe_name', 'ablation_method',
                           'query_train', 'each_feature_value']


        #########################
        args2fname = args.__dict__.copy()
        fname = dict2filename(args2fname, '_', list_args2fname, '', True)

        #########################
        # LOAD ENCODING RESULTS #
        try:
            results, ch_names, args_trf, feature_info = \
                pickle.load(open(os.path.join(args.path2output, fname + '.pkl'), 'rb'))
        except:
            print(f'File not found: {args.path2output}/{fname}.pkl')
            continue
        feature_list = feature_info.keys()

        for i_ch, ch_name in enumerate(ch_names):
            probe_name = get_probe_name(ch_name, args.data_type[0])
            
            if probe_name not in dict_scatter.keys():
                dict_scatter[probe_name] = {}


            for feature in list(feature_list) + ['full']:
                if feature not in dict_scatter[probe_name].keys():
                    dict_scatter[probe_name][feature] = {}
                
                if f'{patient}_{ch_name}' not in dict_scatter[probe_name][feature].keys():
                    dict_scatter[probe_name][feature][f'{patient}_{ch_name}'] = {}
                
                total_score_all_CVs_channels = results[feature]['total_score']
                mean_cv_score = []
                for cv in range(len(total_score_all_CVs_channels)):
                    mean_cv_score.append(total_score_all_CVs_channels[cv][i_ch])
                dict_scatter[probe_name][feature][f'{patient}_{ch_name}'][block] = np.mean(mean_cv_score)
                df = df.append({'probe_name':probe_name,
                                'hemisphere':probe_name[0],
                                'ch_name':ch_name,
                                'patient':patient, 
                                'feature':feature,
                                'r':np.mean(mean_cv_score)}, ignore_index=True)

print(df)
# append rows to an empty DataFrame
fig2, ax_bar = plt.subplots(figsize=(10, 10))
ax_bar = sns.barplot(x="probe_name", y="r", hue="feature", data=df)
fn = f'bar_{args.data_type[0]}_{args.filter[0]}.png'
fig2.savefig(os.path.join(args.path2figures, fn))


#print(dict_scatter)
if not os.path.exists(args.path2figures):
    os.makedirs(args.path2figures)

print('Plotting...')
for probe_name in dict_scatter.keys():
    fig1, ax_scatter = plt.subplots(figsize=(10, 10))
    for feature in list(feature_list) + ['full']:
        ch_names = dict_scatter[probe_name][feature].keys()
        n_points = len(ch_names)
        Xs, Ys = [], []
        for ch_name in ch_names:
            if feature == 'phonology':
                X = dict_scatter[probe_name][feature][ch_name]['auditory']
                Y = dict_scatter[probe_name]['full'][ch_name]['visual'] # Y=0 after subtraction of full below
            elif feature == 'orthography':
                X = dict_scatter[probe_name]['full'][ch_name]['auditory']
                Y = dict_scatter[probe_name][feature][ch_name]['visual']
            else:
                X = dict_scatter[probe_name][feature][ch_name]['auditory']
                Y = dict_scatter[probe_name][feature][ch_name]['visual']
            
            if feature != 'full':
                X = dict_scatter[probe_name]['full'][ch_name]['auditory'] - X
                Y = dict_scatter[probe_name]['full'][ch_name]['visual'] - Y
                color = feature_info[feature]['color']
            else:
                color = 'k'
            Xs.append(X)
            Ys.append(Y)
        if feature == 'semantics':
            color = 'xkcd:orange' 
        #print(Xs, Ys, color)
        ax_scatter.scatter(Xs, Ys, color=color)

    ax_scatter.set_xlabel('Auditory', fontsize=16)
    ax_scatter.set_ylabel('Visual', fontsize=16)
    ax_scatter.set_xlim([-0.5, 1])
    ax_scatter.set_ylim([-0.5, 1])
    ax_scatter.plot([0, 1], [0, 1], transform=ax_scatter.transAxes, ls='--', color='k', lw=2)
    plt.title(f'{probe_name} ({args.data_type[0]}, {args.filter[0]})', fontsize=20)
    fn = f'scatter_{probe_name}_{args.data_type[0]}_{args.filter[0]}.png'
    fig.savefig(os.path.join(args.path2figures, fn))
    print(f'saved to: {args.path2figures}/{fn}')
    plt.close(fig) 
