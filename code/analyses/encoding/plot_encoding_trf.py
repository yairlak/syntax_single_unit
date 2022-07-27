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
from viz import plot_rf_coefs, plot_rf_r2, plot_rf_bar_r2
sys.path.append('..')
from utils.utils import dict2filename
import matplotlib.pyplot as plt
from encoding.models import TimeDelayingRidgeCV


parser = argparse.ArgumentParser(description='Plot TRF results')
# DATA
parser.add_argument('--patient', action='append', default=['505'],
                    help='Patient string')
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike'],
                    action='append', default=['spike'], help='electrode type')
parser.add_argument('--filter', action='append',
                    default=['raw'],
                    help='raw/high-gamma')
parser.add_argument('--smooth', default=50,
                    help='Gaussian smoothing in msec')
parser.add_argument('--probe-name', default=None, nargs='*',
                    action='append', type=str,
                    help='Probe name to plot (ignores channel-name/num)')
parser.add_argument('--channel-name', default=[], nargs='*', action='append',
                    type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=[], nargs='*', action='append',
                    type=int, help='channel number (if empty all channels)')
# MISC
parser.add_argument('--feature-list',
                    nargs='*',
#                    action='append',
                    #default=['position', 'phonemes', 'lexicon'],
                    #default=['position', 'phonology', 'lexicon', 'semantics', 'syntax'],
                    default=['phoneme_pos'],
                    #default='boundaries_phonemes_lexicon_glove_syntax'.split('_'),
                    help='Feature to include in the encoding model')
parser.add_argument('--path2output',
                    default=os.path.join('..', '..', '..',
                                         'Output', 'encoding_models'))
parser.add_argument('--path2figures',
                    default=os.path.join('..', '..', '..',
                                         'Figures', 'encoding_models'))
parser.add_argument('--decimate', default=50, type=float,
                    help='If not empty, decimate data for speed.')
parser.add_argument('--model-type', default='ridge',
                    choices=['ridge', 'lasso', 'ridge_laplacian', 'standard'])
parser.add_argument('--ablation-method', default='remove',
                    choices=['shuffle', 'remove', 'zero'],
                    help='Method used to calcuated feature importance\
                        by reducing/ablating a feature family')
parser.add_argument('--query-train', default="block in [2,4,6] and word_length>1")
parser.add_argument('--query-test', default="block in [2,4,6] and word_length>1")
parser.add_argument('--each-feature-value', default=False, action='store_true',
                    help="Evaluate model after ablating each feature value. \
                         If false, ablate all feature values together")


#############
# USER ARGS #
#############
args = parser.parse_args()
assert len(args.patient) == len(args.data_type) == len(args.filter)
args.patient = ['patient_' + p for p in args.patient]
args.block_type = 'both'
if not args.query_test:
    args.query_test = args.query_train

#args.feature_list = ['is_first_word', 'word_onset']+'positional_orthography_lexicon_syntax'.split('_') 
#args.feature_list = ['is_first_word', 'word_onset']+'positional_orthography'.split('_') 



print('args\n', args)
list_args2fname = ['patient', 'data_type', 'filter', 'decimate', 'smooth',
                   'model_type', 'probe_name', 'ablation_method',
                   'query_train', 'feature_list', 'each_feature_value']
if args.query_train != args.query_test:
    list_args2fname.extend(['query_test'])

if not os.path.exists(os.path.join(args.path2figures, args.patient[0], args.data_type[0])):
    os.makedirs(os.path.join(args.path2figures, args.patient[0], args.data_type[0]))

#########################
args2fname = args.__dict__.copy()
fname = 'TRF_' + dict2filename(args2fname, '_', list_args2fname, '', True)

#########################
# LOAD ENCODING RESULTS #
results, ch_names, args_trf, feature_info = \
    pickle.load(open(os.path.join(args.path2output, 'case_study', 'TRF', fname + '.pkl'), 'rb'))
print(args_trf)
# feature_names = list(feature_info.keys())
# num_features = len(feature_names)

############
# PLOTTING #
############

for i_channel, ch_name in enumerate(ch_names):
    #############
    # PLOT COEF #
    #############
    fig_coef = plot_rf_coefs(results, i_channel, ch_name,
                             feature_info, args, False)
    fname_fig = os.path.join(args.path2figures,
                             args.patient[0],
                             args.data_type[0],
                             'rf_coef_' +
                             fname + f'_{ch_name}.png')
    fig_coef.savefig(fname_fig)
    plt.close(fig_coef)
    print('Figure saved to: ', fname_fig)

    #####################
    # PLOT COEF GROUPED #
    #####################
    fig_coef = plot_rf_coefs(results, i_channel, ch_name,
                             feature_info, args, True)
    fname_fig = os.path.join(args.path2figures, 
                             args.patient[0],
                             args.data_type[0],
                             'rf_coef_' +
                             fname + f'_{ch_name}_groupped.png')
    fig_coef.savefig(fname_fig)
    plt.close(fig_coef)
    print('Figure saved to: ', fname_fig)

    #################
    # PLOT delta R2 #
    #################
    fig_r2 = plot_rf_r2(results, i_channel, ch_name, feature_info, args)
    fname_fig = os.path.join(args.path2figures, 
                              args.patient[0],
                              args.data_type[0],
                              'rf_r_' +
                              fname + f'_{ch_name}_groupped.png')
    fig_r2.savefig(fname_fig)
    plt.close(fig_r2)
    print('Figure saved to: ', fname_fig)

    #################
    # PLOT delta R2 #
    #################
    fig_r2 = plot_rf_bar_r2(results, i_channel, ch_name, feature_info, args)
    fname_fig = os.path.join(args.path2figures, 
                             args.patient[0],
                             args.data_type[0],
                             'rf_bar_r_' +
                             fname + f'_{ch_name}_groupped.png')
    fig_r2.savefig(fname_fig)
    plt.close(fig_r2)
    print('Figure saved to: ', fname_fig)