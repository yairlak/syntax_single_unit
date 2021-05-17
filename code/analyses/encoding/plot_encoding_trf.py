#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:48:59 2021

@author: yl254115
"""
import argparse
import os
import sys
import pickle
from viz import plot_rf_coefs, plot_rf_r2
sys.path.append('..')
from utils.utils import dict2filename
import matplotlib.pyplot as plt
from encoding.models import TimeDelayingRidgeCV

parser = argparse.ArgumentParser(description='Plot TRF results')
# DATA
<<<<<<< HEAD
parser.add_argument('--patient', action='append', default=['502'],
                    help='Patient string')
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike'],
                    action='append', default=['micro'], help='electrode type')
parser.add_argument('--filter', action='append',
                    default=['high-gamma'],
                    help='raw/high-gamma/gaussian-kernel-*')
parser.add_argument('--probe-name', default=['LFSG'], nargs='*',
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
                                         'Figures', 'encoding_models'))
parser.add_argument('--decimate', default=None, type=float,
                    help='If not empty, decimate data for speed.')
parser.add_argument('--model-type', default='ridge',
                    choices=['ridge', 'lasso', 'ridge_laplacian', 'standard'])
parser.add_argument('--ablation-method', default='remove',
                    choices=['shuffle', 'remove', 'zero'],
                    help='Method used to calcuated feature importance\
                        by reducing/ablating a feature family')
parser.add_argument('--query_train', default="block in [1,3,5]")
parser.add_argument('--query_test', default="block in [1,3,5]")
parser.add_argument('--each-feature-value', default=True, action='store_true',
                    help="Evaluate model after ablating each feature value. \
                         If false, ablate all feature values together")
=======
<<<<<<< HEAD
parser.add_argument('--patient', action='append', default=[], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], action='append', default=[], help='electrode type')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'gaussian-kernel-25', 'high-gamma'], action='append', default=[], help='')
parser.add_argument('--probe-name', default=[], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
=======
parser.add_argument('--patient', action='append', default=['479_11'], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], action='append', default=['micro'], help='electrode type')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'gaussian-kernel-25', 'high-gamma'], action='append', default=['gaussian-kernel-10'], help='')
parser.add_argument('--probe-name', default=['LSTG'], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
>>>>>>> b88717b74cc288086bb88dd3d3d33e2d184da968
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
# MISC
parser.add_argument('--path2output', default=os.path.join('..', '..', '..', 'Output', 'encoding_models'), help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--path2figures', default=os.path.join('..', '..', '..', 'Figures', 'encoding_models'), help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--decimate', default=[], type=float, help='If not empty, (for speed) decimate data by the provided factor.')
<<<<<<< HEAD
parser.add_argument('--model-type', default='ridge', choices=['ridge', 'lasso', 'standard']) 
parser.add_argument('--ablation-method', default='remove', choices=['shuffle', 'remove', 'zero'], help='Method used to calcuated feature importance by reducing/ablating a feature family')
parser.add_argument('--query', default=[], help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
=======
parser.add_argument('--model-type', default='ridge_laplacian', choices=['ridge', 'lasso', 'ridge_laplacian', 'standard']) 
parser.add_argument('--ablation-method', default='shuffle', choices=['shuffle', 'remove', 'zero'], help='Method used to calcuated feature importance by reducing/ablating a feature family')
parser.add_argument('--query', default="block in [2,4,6]", help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
>>>>>>> b88717b74cc288086bb88dd3d3d33e2d184da968
>>>>>>> 1c9e1da112fc7bacb6219512afab57bd115e563c

#############
# USER ARGS #
#############
args = parser.parse_args()
assert len(args.patient) == len(args.data_type) == len(args.filter)
args.patient = ['patient_' + p for p in args.patient]
args.block_type = 'both'
if not args.query_test:
    args.query_test = args.query_train
print('args\n', args)
list_args2fname = ['patient', 'data_type', 'filter', 'model_type',
                   'probe_name', 'ablation_method', 'query_train',
                   'each_feature_value']
if args.query_train != args.query_test:
    list_args2fname.extend(['query_test'])

if not os.path.exists(args.path2figures):
    os.makedirs(args.path2figures)

#########################
args2fname = args.__dict__.copy()
fname = dict2filename(args2fname, '_', list_args2fname, '', True)

#########################
# LOAD ENCODING RESULTS #
results, ch_names, args_trf, feature_info = \
    pickle.load(open(os.path.join(args.path2output, fname + '.pkl'), 'rb'))
print(args_trf)
feature_names = list(feature_info.keys())
num_features = len(feature_names)

############
# PLOTTING #
############

for i_channel, ch_name in enumerate(ch_names):
    #############
    # PLOT COEF #
    #############
    fig_coef = plot_rf_coefs(results, i_channel, ch_name,
                             feature_info, args, False)
    fname_fig = os.path.join(args.path2figures, 'rf_coef_' +
                             fname + f'_{ch_name}.png')
    fig_coef.savefig(fname_fig)
    plt.close(fig_coef)
    print('Figure saved to: ', fname_fig)

    #####################
    # PLOT COEF GROUPED #
    #####################
    fig_coef = plot_rf_coefs(results, i_channel, ch_name,
                             feature_info, args, True)
    fname_fig = os.path.join(args.path2figures, 'rf_coef_' +
                             fname + f'_{ch_name}_groupped.png')
    fig_coef.savefig(fname_fig)
    plt.close(fig_coef)
    print('Figure saved to: ', fname_fig)

    #################
    # PLOT delta R2 #
    #################
    fig_r2 = plot_rf_r2(results, i_channel, ch_name, feature_info, args)
    fname_fig = os.path.join(args.path2figures, 'rf_r_' +
                             fname + f'_{ch_name}_groupped.png')
    fig_r2.savefig(fname_fig)
    plt.close(fig_r2)
    print('Figure saved to: ', fname_fig)
<<<<<<< HEAD
=======
    
>>>>>>> 1c9e1da112fc7bacb6219512afab57bd115e563c
