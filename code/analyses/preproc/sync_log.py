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
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--patient', default = '491')
parser.add_argument('--merge-logs', action='store_true', default=True)
args = parser.parse_args()

settings = load_settings_params.Settings('patient_' + args.patient)
nev_folder = op.join(settings.path2patient_folder, 'Raw', 'nev_files')
logs_folder = op.join(settings.path2patient_folder, 'Logs')

fn = f'events_patient_{args.patient}.pkl'
dict_blocks, time0, _ = pickle.load(open(op.join(nev_folder, fn), 'rb'))

fns_logs = sorted(glob.glob(logs_folder + '/*.log'))
assert len(fns_logs) == 6

times_CHEETAH_all, time_stamps_all = [], []
for i_block, fn_log in enumerate(fns_logs):
    if i_block in [0,2,4]:
        target_string = 'DISPLAY_TEXT'
    elif i_block in [1,3,5]:
        target_string = 'AUDIO_PLAYBACK'
    else:
        raise('Wrong block number')
    target_string = 'CHEETAH_SIGNAL'
    time_stamps = np.asarray(list(map(int, 1e6*(dict_blocks[i_block + 1] + time0)))).reshape(-1, 1)  # to MICROSEC
    print(fn_log)
    with open(fn_log, 'r') as f:
        log_lines = f.readlines()
    times_CHEETAH = np.asarray([float(l.split()[0]) for l in log_lines if target_string in l]).reshape(-1, 1) 
    assert time_stamps.size == times_CHEETAH.size
    
    if args.patient == '489' and i_block == 3: #something went wrong with the 4th block of patient 489; remove 3 first triggers
        times_CHEETAH = times_CHEETAH[3:]
        time_stamps = time_stamps[3:]
    if args.patient == '515' and i_block == 0: 
        times_CHEETAH = times_CHEETAH[5:]
        time_stamps = time_stamps[5:]
    model = LinearRegression()
    model.fit(times_CHEETAH, time_stamps)
    r2score = model.score(times_CHEETAH, time_stamps)
    print(f'R^2 log {i_block + 1}: ', r2score)
    times_CHEETAH_all.extend(times_CHEETAH)
    time_stamps_all.extend(time_stamps)
    _, ax = plt.subplots(1)
    ax.scatter(times_CHEETAH, time_stamps)
    ax.set_title(f'block number {i_block+1}: R^2 = {r2score:1.2f}')
    
            
    
    if not args.merge_logs: # extrapolate each log separately and save
        new_log_lines = [' '.join([str(t[0])] +l.split()[1:]) for (t,l) in zip(time_stamps, log_lines)]
        # SAVE
        fn_log_new = op.join(op.dirname(fn_log), f'events_log_in_cheetah_clock_part{i_block+1}.log')
        with open(fn_log_new, 'w') as f:
            [f.write(l+'\n') for l in new_log_lines]
        
if args.merge_logs:
    # RUN REGRESSION MODEL FOR ALL LOGS MERGED TOGETHER
    model = LinearRegression()
    model.fit(times_CHEETAH_all, time_stamps_all)
    r2score = model.score(times_CHEETAH_all, time_stamps_all)
    print('R^2 all logs: ', r2score)
    
    _, ax = plt.subplots(1)
    ax.scatter(times_CHEETAH_all, time_stamps_all)
    ax.set_title(f'All blocks together: R^2 = {r2score:1.2f}')
    
    
    for i_block, fn_log in enumerate(fns_logs):
        with open(fn_log, 'r') as f:
            log_lines = f.readlines()
        new_log_lines = []
        for l in log_lines:
            t = float(l.split()[0])
            l_end = ' '.join(l.split()[1:])
            t_synced = int(model.predict(np.asarray([t]).reshape(1, -1))[0,0])
            new_log_lines.append(f'{t_synced} {l_end}')
                
        # SAVE
        fn_log_new = op.join(op.dirname(fn_log), f'events_log_in_cheetah_clock_part{i_block+1}.log')
        with open(fn_log_new, 'w') as f:
            [f.write(l+'\n') for l in new_log_lines]