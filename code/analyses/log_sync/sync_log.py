#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 21:16:50 2021

@author: yl254115
"""

import sys, os, argparse
sys.path.append('..')
import os.path as op

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
from misc import remove_outliers, load_microphone_data, load_auditory_stimulus, find_max_cross_corr_microphone, plot_cross_correlation
from pprint import pprint
from data_manip import read_events, read_logs

parser = argparse.ArgumentParser()
<<<<<<< HEAD
parser.add_argument('--patient', default = '553')
=======
parser.add_argument('--patient', default = '545')
>>>>>>> d6d2a703505d03bd6159975eb2f4db1b9c6fef66
parser.add_argument('--recording-system', choices=['Neuralynx', 'BlackRock'], default='Neuralynx')
parser.add_argument('--IXs-block-logs', default=[0,1,2,3,4,5], help='Since there could be more cheetah logs than block, these indexes define the log indexes of interest')
parser.add_argument('--dt', default = 5, help='size of half window for cross-correlation in seconds')
parser.add_argument('--refine-with-mic', action='store_true', default=False)
parser.add_argument('--merge-logs', action='store_true', default=True)
parser.add_argument('--viz', action='store_true', default=True)
parser.add_argument('-v', '--verbose', action='store_true', default=True)
args = parser.parse_args()
if isinstance(args.IXs_block_logs, str):
    args.IXs_block_logs = eval(args.IXs_block_logs)
pprint(args)

logs_folder = os.path.join('..', '..', '..', 'Data', 'UCLA',
                           'patient_' + args.patient, 'Logs')

#################
# Read NEV file #
#################

time_stamps, event_nums_zero, sfreq = read_events(args)
print(f'sfreq = {sfreq}')

#if args.patient in ['541']:
#    time_stamps = [0, 98.5, 99, 99.1, 99.5, 99.6] + time_stamps
#    event_nums_zero = [100, 0, 1, 0, 2, 0] + event_nums_zero

# if args.patient in ['544']:
#     time_stamps = [66.3, 66.3, 66.8, 66.8, 67.36, 67.36] + time_stamps
#     event_nums_zero = [100, 0, 1, 0, 2, 0] + event_nums_zero


# Plot TTLs
fig, ax = plt.subplots()
ax.plot(time_stamps, event_nums_zero, color='b')
ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Event ID', fontsize=16)
dir_figures = op.join('..', '..', '..',
                      'Figures', 'log_sync', f'patient_{args.patient}')
os.makedirs(dir_figures, exist_ok=True)
fn_fig = op.join(dir_figures, f'events_pt_{args.patient}.png')
plt.savefig(fn_fig)
plt.close(fig)

####################################################
# READ LOGS AND KEEP ONLY THOSE WITH SENT TRIGGERS #
####################################################

<<<<<<< HEAD
if args.patient in ['539', '541', '544', '551', '553', '554_4']:
=======
if args.patient in ['539', '541', '544', '545', '551', '552', '553', '554_4', '554_13']:
>>>>>>> d6d2a703505d03bd6159975eb2f4db1b9c6fef66
    start_event = 3 # First three triggers are missing
    missing_first_events = 3
else:
    start_event = 100
    missing_first_events = 0
    
dict_events = read_logs(time_stamps, event_nums_zero, args,
                        start_event=start_event,
                        missing_first_events=missing_first_events)

##################################
# REGRESS EVENT ON CHEETAH TIMES #
##################################

# RUN REGRESSION FIRST FOR ALL LOGS MERGED TOGETHER
times_log_all, time_stamps_all = [], []
for i_log in dict_events.keys():
    times_log = dict_events[i_log]['times_log']
    times_device = dict_events[i_log]['times_device']
    times_log, times_device = remove_outliers(times_log, times_device, i_log, args)
    times_log_all.extend(times_log)
    time_stamps_all.extend(times_device)
model_all = LinearRegression()

print(len(times_log_all), len(time_stamps_all))
assert len(times_log_all) > 0 and len(time_stamps_all) > 0
model_all.fit(times_log_all, time_stamps_all)
r2score_all = model_all.score(times_log_all, time_stamps_all)
print('R^2 all logs: ', r2score_all)
_, ax = plt.subplots(1)
ax.scatter(times_log_all, time_stamps_all)
ax.set_title(f'All logs together: R^2 = {r2score_all:1.2f}')

# REGRESSION FOR EACH LOG
cnt_log = 0
for i_log in dict_events.keys():
    print(dict_events[i_log]['log_filename'])
    times_log = dict_events[i_log]['times_log']
    times_device = dict_events[i_log]['times_device']
    lines_log = dict_events[i_log]['lines_log']
    times_log, times_device = remove_outliers(times_log, times_device, i_log, args)
    
    model = LinearRegression().fit(times_log, times_device)
    r2score = model.score(times_log, times_device)
    print(f'R^2 log IX = {i_log}: ', r2score)
   
    if i_log in args.IXs_block_logs:
        fig, ax = plt.subplots(1)
        ax.scatter(times_log, times_device)
        ax.set_title(f'log number {i_log+1}: R^2 = {r2score:1.5f}')
        ax.plot(times_log, model.intercept_[0] + model.coef_[0] * times_log, ls='--', color='k', lw=2)
        ax.set_xlabel('Time (log)', fontsize=16)
        ax.set_ylabel('Time (recording-device)', fontsize=16)
        fn_fig = fn_fig = op.join(dir_figures, f'regrssion_log2device_pt_{args.patient}_block_{cnt_log+1}.png')
        plt.savefig(fn_fig)
        plt.close(fig)
        new_log_lines = []
        for l in dict_events[i_log]['lines_log']:
            t = float(l.split()[0])
            l_end = ' '.join(l.split()[1:])
            if args.merge_logs:
                t_regress = int(model_all.predict(np.asarray([t]).reshape(1, -1))[0]) # microsec
            else:
                t_regress = int(model.predict(np.asarray([t]).reshape(1, -1))[0]) # microsec
            if args.refine_with_mic and 'AUDIO_PLAYBACK_ONSET' in l:
                t_regress_sec = t_regress/1e6 # time0 is in sec
                fn_wav = l.split()[-1]
                # print(f'Cross-correlating with wav file {fn_wav}')
                # LOAD MIC DATA AND CROP IT BASED ON ESTIMATED TIME FROM REGRESSION
                mic_data, sfreq_mic, first_sample_in_window = load_microphone_data(t_regress_sec, args, dt=args.dt)
                # LOAD WAV DATA AND DOWNSAMPLE TO MIC SFREQ
                wav_downsampled, sfreq_down = load_auditory_stimulus(fn_wav, sfreq_mic, args)
                wav_downsampled = np.concatenate((np.zeros(int(sfreq_down)), wav_downsampled, np.zeros(int(sfreq_down))))
                # calc onset times wrt first_sample_in_window, using cross-correlation
                t_mic_waveform, t_mic_spect, t_both, xcorr_wavform, xcorr_filt, xcorr_both = find_max_cross_corr_microphone(mic_data, wav_downsampled, sfreq_mic)
                # Add 1sec due to silence padded before wav file
                t_mic_waveform += 1
                t_mic_spect += 1 
                t_both += 1
                # if np.abs(t_mic_sec - t_estimated_sec) > 0.1: # more than 100ms difference()
                #     print(f'Warning: large difference between estimated and mic time {fn_wav}: mic - {t_mic_sec:1.2f} regression - {t_estimated_sec:1.2f} ({np.abs(t_mic_sec - t_estimated_sec):1.2f})')    
                if args.viz:
                    fig_waveforms = plot_cross_correlation(mic_data, wav_downsampled, sfreq, xcorr_wavform, xcorr_filt, xcorr_both, first_sample_in_window, args, dt=args.dt)
                    fn_fig = f'cross_correlation_waveforms_pt_{args.patient}_{os.path.basename(fn_wav)}_log_{i_log+1}.png'
                    fig_waveforms.savefig(os.path.join(dir_figures, fn_fig))
                    plt.close(fig_waveforms)
                    
                t1 = f'{int(np.floor(t_regress_sec/60))}:{t_regress_sec%60}'
                t2 = f'{int(np.floor((t_mic_spect + first_sample_in_window/sfreq)/60))}:{(t_mic_spect + first_sample_in_window/sfreq)%60}'
                print(t1, t2)
                t_estimated = (t_mic_waveform + first_sample_in_window/sfreq)*1e6 # from sec to microsec
            else:
                t_estimated = t_regress
            
            new_log_lines.append(f'{t_estimated} {l_end}')
        # SAVE
        fn_log_new = op.join(logs_folder, f'events_log_in_cheetah_clock_part{cnt_log+1}.log')
        with open(fn_log_new, 'w') as f:
            [f.write(l+'\n') for l in new_log_lines]
        cnt_log += 1


