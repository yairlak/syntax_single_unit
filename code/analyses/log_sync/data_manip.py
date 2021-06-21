#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:19:24 2021

@author: yl254115
"""

import sys, os, argparse, glob
sys.path.append('..')
from utils import load_settings_params
from neo import io
import numpy as np
import scipy.io as sio


def read_events(args):
    settings = load_settings_params.Settings('patient_' + args.patient)
    params = load_settings_params.Params('patient_' + args.patient)
    session_folder = os.path.join('..', settings.path2patient_folder, 'Raw', 'nev_files')
    
    nev_files = glob.glob(os.path.join(session_folder, 'Events.*'))
    assert len(nev_files) > 0
    
    event_nums_zero, time_stamps, IXs2nev = [], [], []
    duration_prev_nevs = 0 # For blackrock: needed to concat nev files. Adds the duration of the previous file(s)
    for i_nev, nev_file in enumerate(sorted(nev_files)):
        print(f'Reading {nev_file}')
        if nev_file[-3:] == 'nev':
            print('nev file')
            if args.recording_system == 'Neuralynx':
                reader = io.NeuralynxIO(session_folder)
                blks = reader.read(lazy=False)
                sfreq = params.sfreq_raw # FROM NOTES
                time0, timeend = reader.global_t_start, reader.global_t_stop
                
                events_times, events_ids = [], []
                for segment in blks[0].segments:
                    event_times_mat = segment.events
                    for neo_event_times in event_times_mat:
                        ttl = int(neo_event_times.name.split('ttl=')[-1])
                        #if IX2event_id[IX] in event_id.values():
                        times = np.asarray(neo_event_times.times)
                        events_times.extend(times) # in SECONDS
                        events_ids.extend([ttl] * len(times))
                events_ids = np.asarray(events_ids) 
                events_times = np.asarray(events_times)
                print(f'Number of events in nev {nev_file}: {len(events_times)}')
                IX_chrono = events_times.argsort()
                time_stamps.extend(events_times[IX_chrono])
                event_nums_zero.extend(events_ids[IX_chrono])
                del reader, blks, segment
            elif args.recording_system == 'BlackRock':
                reader = io.BlackrockIO(nev_file)
                time0, timeend = reader._seg_t_starts[0], reader._seg_t_stops[0]
                sfreq = reader.header['unit_channels'][0][-1] # FROM FILE
                events = reader.nev_data['NonNeural'][0]
                events_times = duration_prev_nevs + np.asarray([float(e[0]/sfreq) for e in events])
                time_stamps.extend(events_times)
                event_nums = [e[4] for e in events] 
                event_nums_zero.extend(event_nums - min(event_nums))
                print('time0, timeend = ', time0, timeend)
        elif nev_file[-3:] == 'mat':
            assert len(nev_files) == 1
            print('mat file')
            events = loadmat(nev_file)
            if 'timeStamps' in events:
                time_stamps = events['timeStamps'][0, :]
                event_nums_zero = event_nums = events['TTLs'][0, :]
            else:
                time_stamps = events['NEV']['Data']['SerialDigitalIO']['TimeStampSec']
                event_nums_zero = event_nums = events['NEV']['Data']['SerialDigitalIO']['UnparsedData']
                #print(time_stamps)

            # get time0, timeend and sfreq from ncs files
            if args.recording_system == 'Neuralynx':
                reader = io.NeuralynxIO(session_folder)
                sfreq = reader._sigs_sampling_rate
                time0, timeend = reader.global_t_start, reader.global_t_stop
            elif args.recording_system == 'BlackRock':
                nev_files = glob.glob(os.path.join(session_folder, '*.nev'))
                reader = io.BlackrockIO(nev_files[0])
                time0, timeend = reader._seg_t_starts[0], reader._seg_t_stops[0]
                sfreq = reader.header['unit_channels'][0][-1] # FROM FILE
        else:
            raise(f'Unrcognized event file: {nev_file}')
        if timeend:
            duration_prev_nevs += timeend
    assert len(event_nums_zero) == len(time_stamps)
    
    return time_stamps, event_nums_zero, time0, timeend, sfreq


def read_logs(time_stamps, event_nums_zero, time0, args):
    
    
    settings = load_settings_params.Settings('patient_' + args.patient)
    params = load_settings_params.Params('patient_' + args.patient)
    logs_folder = os.path.join(settings.path2patient_folder, 'Logs')
    
    dict_events = {}
    IX_time_stamps = 0
    cnt_log = 0
    fns_logs = sorted(glob.glob(os.path.join('..', logs_folder, 'events_log_????-??-??_??-??-??.log')))
    fns_logs_with_CHEETAH = []
    num_triggers_per_log = []
    for fn_log in fns_logs:
        print(fn_log)
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
                    curr_time = time_stamps[IX_time_stamps]
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
                
            if cnt_log == 1 and args.patient == '496':
                times_device = np.asarray(times_device) + 30
            dict_events[cnt_log]['times_device'] = np.asarray(list(map(int, 1e6*(np.asarray(times_device) + time0)))).reshape(-1, 1)  # to MICROSEC
            dict_events[cnt_log]['IXs2event_nums_zero'] = np.asarray(IXs2event_nums_zero)
            #print(dict_events[cnt_log]['IXs2event_nums_zero'])
            assert dict_events[cnt_log]['times_device'].size == dict_events[cnt_log]['times_log'].size
            cnt_log += 1
    return dict_events


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict
