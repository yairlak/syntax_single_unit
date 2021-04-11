#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 21:16:50 2021

@author: yl254115
"""

import sys, argparse, glob
sys.path.append('..')
from utils import load_settings_params
import os.path as op
from neo import io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
from misc import remove_outliers, refine_with_microphone

parser = argparse.ArgumentParser()
parser.add_argument('--patient', default = '491')
parser.add_argument('--recording-system', choices=['Neuralynx', 'BlackRock'], default='Neuralynx')
parser.add_argument('--IXs-block-logs', default=[0,1, 2,3,4,5], help='Since there could be more cheetah logs than block, these indexes define the log indexes of interest')
parser.add_argument('--refine-with-mic', action='store_true', default=True)
parser.add_argument('--merge-logs', action='store_true', default=False)
parser.add_argument('-v', '--verbose', action='store_true', default=False)
args = parser.parse_args()

settings = load_settings_params.Settings('patient_' + args.patient)
nev_folder = op.join(settings.path2patient_folder, 'Raw', 'nev_files')
logs_folder = op.join(settings.path2patient_folder, 'Logs')
session_folder = op.join(settings.path2patient_folder, 'Raw', 'nev_files')
params = load_settings_params.Params('patient_' + args.patient)

#################
# Read NEV file #
#################

nev_files = glob.glob(session_folder + '/*.nev')
# assert len(nev_files) == 1
# nev_file = nev_files[0]
event_nums_zero, time_stamps, IXs2nev = [], [], []
# nev_files.reverse()
_, ax = plt.subplots()
colors = ['r', 'g', 'b']
duration_prev_nevs = 0 # For blackrock: needed to concat nev files. Adds the duration of the previous file(s)
for i_nev, nev_file in enumerate(sorted(nev_files)):
    print(f'Reading {nev_file}')
    if args.recording_system == 'Neuralynx':
        reader = io.NeuralynxIO(session_folder)
        blks = reader.read(lazy=False)
        #print('Sampling rate of signal:', reader._sigs_sampling_rate)
        sfreq = params.sfreq_raw # FROM NOTES
        time0, timeend = reader.global_t_start, reader.global_t_stop
        internal_event_ids = reader.internal_event_ids
        IX2event_id = {IX:e_id for IX, (x, e_id) in enumerate(internal_event_ids)}
        
        events_times, events_ids = [], []
        for segment in blks[0].segments:
            event_times_mat = segment.events
            for IX, times in enumerate(event_times_mat):
                #if IX2event_id[IX] in event_id.values():
                events_times.extend(times) # in SECONDS
                events_ids.extend([IX2event_id[IX]] * len(times))
        events_ids = np.asarray(events_ids) 
        events_times = np.asarray(events_times)
        print(f'Number of events in nev {nev_file}: {len(events_times)}')
        IX_chrono = events_times.argsort()
        time_stamps.extend(events_times[IX_chrono])
        event_nums_zero.extend(events_ids[IX_chrono])
        ax.plot(events_times[IX_chrono], events_ids[IX_chrono], color=colors[i_nev])
        print('time0, timeend = ', time0, timeend)
        del reader, blks, segment
    elif args.recording_system == 'BlackRock':
        reader = io.BlackrockIO(nev_file)
        time0, timeend = reader._seg_t_starts[0], reader._seg_t_stops[0]
        #sfreq = params.sfreq_raw # FROM NOTES
        sfreq = reader.header['unit_channels'][0][-1] # FROM FILE
        events = reader.nev_data['NonNeural'][0]
        events_times = duration_prev_nevs + np.asarray([float(e[0]/sfreq) for e in events])
        time_stamps.extend(events_times)
        event_nums = [e[4] for e in events] 
        event_nums_zero.extend(event_nums - min(event_nums))
        ax.plot(events_times, event_nums - min(event_nums), color=colors[i_nev])
    duration_prev_nevs += timeend
assert len(event_nums_zero) == len(time_stamps)

####################################################
# READ LOGS AND KEEP ONLY THOSE WITH SENT TRIGGERS #
####################################################

dict_events = {}
IX_time_stamps = 0
cnt_log = 0
fns_logs = sorted(glob.glob(logs_folder + '/events_log_????-??-??_??-??-??.log'))
fns_logs_with_CHEETAH = []
num_triggers_per_log = []
for fn_log in fns_logs:
    with open(fn_log, 'r') as f:
        lines_log = f.readlines()
    str_CHEETAH = 'CHEETAH_SIGNAL SENT_AFTER_TIME'
    times_log = np.asarray([float(l.split()[0]) for l in lines_log if str_CHEETAH in l]).reshape(-1, 1)
    num_triggers = len(times_log)    
    if num_triggers>0:
        dict_events[cnt_log] = {}
        dict_events[cnt_log]['log_filename'] = fn_log
        dict_events[cnt_log]['lines_log'] = lines_log
        dict_events[cnt_log]['num_triggers'] = num_triggers
        dict_events[cnt_log]['times_log'] = times_log
        # FIND EVENTS TIMES        
        times_device, IXs2event_nums_zero = [], []
        next_expected_event = 100
        for i_trigger in range(num_triggers):
            curr_event, curr_time = event_nums_zero[IX_time_stamps], time_stamps[IX_time_stamps]
            while curr_event != next_expected_event: # roll array until next expected event arrives
                cur_time = time_stamps[IX_time_stamps]
                last_event, last_time = curr_event, curr_time
                IX_time_stamps += 1
                curr_event, curr_time = event_nums_zero[IX_time_stamps], time_stamps[IX_time_stamps]
                # if next expected arrives but looks like a false trigger then roll one more
                last_expected_event = (next_expected_event-1)
                if last_expected_event == 0: last_expected_event =100
                if curr_event == next_expected_event and last_event == last_expected_event and (curr_time - last_time) < 0.05: #threshold in sec
                    # IX_time_stamps += 1
                    curr_event = event_nums_zero[IX_time_stamps]
            if args.verbose: print(cnt_log, num_triggers, i_trigger, IX_time_stamps, curr_event, next_expected_event)
            IXs2event_nums_zero.append(IX_time_stamps)
            times_device.append(time_stamps[IX_time_stamps])
            next_expected_event = (next_expected_event % 100) + 1
            
            
        dict_events[cnt_log]['times_device'] = np.asarray(list(map(int, 1e6*(np.asarray(times_device) + time0)))).reshape(-1, 1)  # to MICROSEC
        dict_events[cnt_log]['IXs2event_nums_zero'] = np.asarray(IXs2event_nums_zero)
        print(dict_events[cnt_log]['IXs2event_nums_zero'])
        assert dict_events[cnt_log]['times_device'].size == dict_events[cnt_log]['times_log'].size
        cnt_log += 1

##################################
# REGRESS EVENT ON CHEETAH TIMES #
##################################
cnt_log = 0
times_log_all, time_stamps_all = [], []
for i_log in dict_events.keys():
    print(dict_events[i_log]['log_filename'])
    
    times_log = dict_events[i_log]['times_log']
    times_device = dict_events[i_log]['times_device']
    lines_log = dict_events[i_log]['lines_log']
    
    
    times_log, times_device = remove_outliers(times_log, times_device, i_log, args)
    
    model = LinearRegression().fit(times_log, times_device)
    r2score = model.score(times_log, times_device)
    print(f'R^2 log {i_log + 1}: ', r2score)
    times_log_all.extend(times_log)
    time_stamps_all.extend(times_device)
    _, ax = plt.subplots(1)
    ax.scatter(times_log, times_device)
    ax.set_title(f'log number {i_log+1}: R^2 = {r2score:1.2f}')
    
    if not args.merge_logs: # extrapolate each log separately and save
        if i_log in args.IXs_block_logs:
            new_log_lines = []
            for l in dict_events[i_log]['lines_log']:
                t = float(l.split()[0])
                l_end = ' '.join(l.split()[1:])
                t_synced = int(model.predict(np.asarray([t]).reshape(1, -1))[0])
                if args.refine_with_mic and 'AUDIO_PLAYBACK_ONSET' in l:
                    fn_wav = l.split()[-1]
                    # print(f'Cross-correlating with wav file {fn_wav}')
                    t_mic_sec = refine_with_microphone(t_synced/1e6 - time0, fn_wav, args, dt=2, viz=True)
                    t_synced = (t_mic_sec + time0)*1e6
                new_log_lines.append(f'{t_synced} {l_end}')
            # SAVE
            fn_log_new = op.join(op.dirname(fn_log), f'events_log_in_cheetah_clock_part{cnt_log+1}.log')
            with open(fn_log_new, 'w') as f:
                [f.write(l+'\n') for l in new_log_lines]
            cnt_log += 1

#####################
# GENERATE NEW LOGS #
#####################    
if args.merge_logs:
    # RUN REGRESSION MODEL FOR ALL LOGS MERGED TOGETHER
    model = LinearRegression()
    model.fit(times_log_all, time_stamps_all)
    r2score = model.score(times_log_all, time_stamps_all)
    print('R^2 all logs: ', r2score)
    
    _, ax = plt.subplots(1)
    ax.scatter(times_log_all, time_stamps_all)
    ax.set_title(f'All logs together: R^2 = {r2score:1.2f}')
    
    # ROBUST REGRESSION
    # huber = HuberRegressor().fit(times_log, times_device)
    # r2score_huber = huber.score(times_log, times_device)
    # print(f'R^2 (Huber) log {i_log + 1}: ', r2score_huber)
    
    
    cnt_log = 0
    for i_log in dict_events.keys():
        if i_log in args.IXs_block_logs:
            fn_log = dict_events[i_log]['log_filename']
            lines_log = dict_events[i_log]['lines_log']
            new_log_lines = []
            for l in lines_log:
                t = float(l.split()[0])
                l_end = ' '.join(l.split()[1:])
                t_synced = int(model.predict(np.asarray([t]).reshape(1, -1))[0])
                new_log_lines.append(f'{t_synced} {l_end}')
                    
            # SAVE
            fn_log_new = op.join(op.dirname(fn_log), f'events_log_in_cheetah_clock_part{cnt_log+1}.log')
            with open(fn_log_new, 'w') as f:
                [f.write(l+'\n') for l in new_log_lines]
            cnt_log += 1