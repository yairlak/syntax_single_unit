#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:48:59 2021

@author: yl254115
"""

import argparse, os, sys, pickle, glob
import mne
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
from utils.read_logs_and_features import extend_metadata
from utils.utils import dict2filename
from utils.features import get_features
import numpy as np
from nilearn import plotting  

parser = argparse.ArgumentParser(description='Train an encoding model on neural data')
# DATA
parser.add_argument('--patient', action='append', default=[], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], default='micro', help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='word', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'gaussian-kernel-25', 'high-gamma'], default='gaussian-kernel', help='')
parser.add_argument('--probe-name', default=[], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channe-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
# QUERY
parser.add_argument('--query', default='word_length>1 and (block in [2, 4, 6])', help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--feature', default='word_length')
parser.add_argument('--block-type', choices=['auditory', 'visual', 'both'], default='both', help='Block type will be added to the query in the comparison')
# MODEL
parser.add_argument('--model-type', default='ridge', choices=['ridge', 'lasso', 'standard']) 
# MISC
parser.add_argument('--tmin', default=-0.1, type=float, help='')
parser.add_argument('--tmax', default=0.7, type=float, help='')
parser.add_argument('--decimate', default=[], type=float, help='If not empty, (for speed) decimate data by the provided factor.')
parser.add_argument('--path2output', default=os.path.join('..', '..', '..', 'Output', 'encoding_models'), help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")


#############
# USER ARGS #
#############
args = parser.parse_args()
args.patient = ['patient_' + p for p in  args.patient]
#if not args.probe_name:
#    args.probe_name = ['All']
print('args\n', args)
assert len(args.patient)==len(args.probe_name)
# FNAME 
#list_args2fname = ['patient', 'data_type', 'filter', 'level', 'block_type', 'model_type', 'ch_name', 'feature_list', 'query']
list_args2fname = ['patient', 'data_type', 'filter', 'level', 'block_type', 'model_type', 'ch_name', 'query']


np.random.seed(1)

#################
# LOAD METADATA #
#################
epochs = mne.read_epochs('../../../Data/UCLA/patient_479_11/Epochs/patient_479_11_micro_gaussian-kernel_word-epo.fif', preload=False)
metadata = epochs.metadata
metadata = extend_metadata(metadata)
_, feature_values, feature_info, feature_groups = get_features(metadata, []) # GET DESIGN MATRIX
feature_info['full'] = {}
feature_info['full']['color'] = 'k'

#############################
# Get elec locations in MNI #
#############################
with open('../../../Data/UCLA/MNI_coordinates.txt') as f:
    elec_locations = f.readlines()

elec_locations_dict = {}
for l in elec_locations:
    loc = l.split(',')[0]
    x = float(l.split(',')[1])
    y = float(l.split(',')[2])
    z = float(l.split(',')[3])
    elec_locations_dict[loc] = (x, y, z)


#########################
# LOAD ENCODING RESULTS #
#########################

names_from_all_patients = []
dmn_coords = []
colors = []
sizes = []
for patient, probes in zip(args.patient, args.probe_name):
    for probe in probes:
        #for ch_name in epochs.ch_names:
        for ch_num in range(1, 9):
            print(patient, probe, ch_num)
            args.ch_name = f'*{probe}{ch_num}'
            args2fname = args.__dict__.copy()
            args2fname['patient'] = patient
            args2fname['probe_name'] = probe
            fname = dict2filename(args2fname, '_', list_args2fname, '', True)
            found_fnames = glob.glob(os.path.join(args.path2output,fname[:90] + '*.pkl'))
            if len(found_fnames) == 1:
                with open(found_fnames[0], 'rb') as f:
                    model, scores, args_encoding = pickle.load(f)
                scores_full = np.asarray(scores['full']['mean'])
                r2_full_mean = np.mean(scores_full, axis=1)
                if args.feature != 'full':
                    size_scale = 15
                    scores_reduced = np.asarray(scores[args.feature]['mean'])
                    r2_reduced_mean = np.mean(scores_reduced, axis=1)
                    effect_size_mean =  np.clip(r2_full_mean, 0, None) - np.clip(r2_reduced_mean, 0, None)
                else:
                    size_scale = 15
                    effect_size_mean =  r2_full_mean
                max_value_acorss_time = np.max(effect_size_mean)
                max_value_acorss_time = np.max([max_value_acorss_time, 0])
                if probe in elec_locations_dict.keys():
                    coords = elec_locations_dict[probe] + 3*np.random.rand(1, 3) # add some jitter
                    dmn_coords.append(coords)
                    colors.append(feature_info[args.feature]['color'])
                    sizes.append(size_scale*max_value_acorss_time)
                else:
                    print('probe name not in elec location list: %s %s:' % (probe, patient))
            else:
                pass
                #print(f'File not found for {os.path.join(args.path2output,fname[:95])}')
print(np.max(sizes))
args2fname = args.__dict__.copy()
fname = dict2filename(args2fname, '_', list_args2fname + ['feature'], '', True)
num_points = len(dmn_coords)

view = plotting.plot_connectome(np.zeros((num_points, num_points)),                               
                                          np.squeeze(np.asarray(dmn_coords)),
                         title=f'{args.feature} {args.query}', node_color = colors, node_size = sizes,
                         display_mode='lyrz', output_file = os.path.join('../../../Figures/encoding_models/', fname + '.png'))


print(f"HTML saved to: {os.path.join('../../../Figures/encoding_models', fname.png)}")


            
            
