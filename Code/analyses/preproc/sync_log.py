#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 21:16:50 2021

@author: yl254115
"""

import sys, pickle, argparse, glob
sys.path.append('..')
from utils import load_settings_params
import os.path as op
from neo import io
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--patient', default = '530')
parser.add_argument('--recording-system', choices=['Neuralynx', 'BlackRock'], default='BlackRock')
args = parser.parse_args()

settings = load_settings_params.Settings('patient_' + args.patient)
nev_folder = op.join(settings.path2patient_folder, 'Raw', 'nev_files')
logs_folder = op.join(settings.path2patient_folder, 'Logs')

fn = f'events_patient_{args.patient}.pkl'
dict_blocks = pickle.load(open(op.join(nev_folder, fn), 'rb'))

fns_logs = sorted(glob.glob(logs_folder + '/*.log'))
assert len(fns_logs) == 6

for i_block, fn_log in enumerate(fns_logs):
    if i_block in [0,2,4]:
        target_string = 'DISPLAY_TEXT'
    elif i_block in [1,3,5]:
        target_string = 'AUDIO_PLAYBACK'
    else:
        raise('Wrong block number')
    time_stamps = dict_blocks[i_block + 1]
    print(fn_log)
    with open(fn_log, 'r') as f:
        log_lines = f.readlines()
    log_lines = [l for l in log_lines if target_string in l]
    assert len(time_stamps) == len(log_lines)
    new_log_lines = [' '.join([str(t)] +l.split()[1:]) for (t,l) in zip(time_stamps, log_lines)]
    
    # SAVE
    fn_log_new = op.join(op.dirname(fn_log), f'events_log_in_cheetah_clock_part{i_block+1}.log')
    with open(fn_log_new, 'w') as f:
        [f.write(l+'\n') for l in new_log_lines]