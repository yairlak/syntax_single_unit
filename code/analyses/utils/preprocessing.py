#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:04:33 2022

@author: yair
"""

import numpy as np
import scipy.signal
from mne.filter import filter_data
from scipy import fftpack

LINE_NOISE = [50, 60]

def preprocess_data(data, frequency_bands={}, reference=None):

    data = clean_data(data)

    # Reference electrodes
    if reference != None:
        data = apply_reference(data, reference)
        
    # Filter data
    if any(frequency_bands):
        data = apply_filters(data, frequency_bands)

    return data


def hilbert3(x):
    return scipy.signal.hilbert(x, 
                fftpack.next_fast_len(len(x)), 
                axis=0)[:len(x)]

def laplacian_reference(seeg, channels):
    seeg_ref = seeg.copy()
    electrodes = np.unique([ch.rstrip('0123456789') for ch in channels
                                                    if '+' not in ch])
    for electrode in electrodes:
        electrode_channels = [ch for ch in channels if electrode==ch.rstrip('0123456789')]
        adjacent_channels = []
        for i, ch in enumerate(electrode_channels):
            current_ch = channels.index(electrode_channels[i])
            if i==0:
                adjacent_channels = [channels.index(electrode_channels[i+1])]
            elif i==(len(electrode_channels)-1):
                adjacent_channels = [channels.index(electrode_channels[i-1])]
            else:
                adjacent_channels = [channels.index(electrode_channels[i-1]),
                                     channels.index(electrode_channels[i+1])]
            ch_average = np.mean([seeg[:, ch] for ch in adjacent_channels])
            seeg_ref[:, current_ch] = seeg[:, current_ch] - ch_average
    return seeg_ref

def clean_data(data, cutoff_l=0.5, cutoff_h=None):
    data['eeg'] = scipy.signal.detrend(data['eeg'], axis=0)

    print('Filtering data from {} to {}'.format(cutoff_l, cutoff_h))
    data['eeg'] = filter_data(data['eeg'].T, data['fs'],
                           cutoff_l, cutoff_h, verbose=0).T
    return data

def common_electrode_reference(seeg, channels):
    seeg_ref = seeg.copy()
    electrodes = np.unique([ch.rstrip('0123456789') for ch in channels])

    for electrode in electrodes:
        electrode_channels = [channels.index(ch) for ch in channels if electrode in ch]
        seeg_ref[:, electrode_channels] = np.subtract(seeg_ref[:, electrode_channels], 
                                                      np.mean(seeg[:, electrode_channels],
                                                              axis=1, keepdims=1))

    return seeg_ref

def common_average_reference(seeg, channels):
    seeg_ref = seeg.copy()
    idc = [channels.index(ch) for ch in channels]
    seeg_ref[:, idc] = seeg_ref[:, idc] - np.mean(seeg_ref[:, idc], axis=1, keepdims=1)
    return seeg_ref

def apply_reference(data, reference_type='cer',
                    exlude_chs=['EKG+', 'MKR2+']):
    ''' 
    Laplacian:
        For each electrode:
            1) Retrieve 2 adjacent electrodes
            2) Remove the average
    '''
    reference_type = reference_type.lower()
    channels = data['channel_names'].copy() # Can't remove chs here, because it will change the indices, specifically for CAR
    # channels = [ch for ch in data['channel_names'] if ch not in exlude_chs]

    if reference_type == 'laplacian':
        seeg = laplacian_reference(data['eeg'], channels)
    elif reference_type == 'cer':
        seeg = common_electrode_reference(data['eeg'], channels)
    elif reference_type == 'car':
        seeg = common_average_reference(data['eeg'], channels)

    data['eeg'] = seeg

    return data

def get_line_noise_filters(band, fs, line_frequencies, frequency_offset=2):
    # Finds all possible harmonics and returns all the corresponding 
    # notch filters with frequency_offset.
    line_noise_filters = []
    for line_frequency in line_frequencies:
        line_harmonics = [line_frequency*i for i in range(int(fs/2/line_frequency))]
        line_harmonics = np.array(line_harmonics)
        harmonics = np.where((line_harmonics>band[0]) & (line_harmonics<band[1]))[0]
        line_noise_filters += [[line_frequency*h+frequency_offset, \
                                line_frequency*h-frequency_offset] for h in harmonics]
    return line_noise_filters

def filter_eeg(data, 
               band,
               line_freq=[50, 60]):
    eeg = data['eeg'].copy()

    # Filter
    # Check for Nyquist-Shannon Theorem
    if any(f >= (data['fs']/2) for f in band):
        print("Value within band {} doesn't adhere to Nyquist-Shannon theorem with Fs={}. Returning None..." \
                .format(band, data['fs']))
        return
    
    print('Filtering bandpass: {}'.format(band))

    # Filter eeg
    eeg = filter_data(eeg.T, data['fs'],
                      band[0], band[1],
                       method='iir',
                      verbose=0).T
    
    # Filter for line noise and its harmonics if neccesary
    line_noise_filters = get_line_noise_filters(band, data['fs'], line_freq)
    for noise_filter in line_noise_filters:
        # Band-Stop filter. Adjust offset keyword argument of
        # get_line_noise_filters() to set the width of the
        # filter.
        eeg = filter_data(eeg.T, data['fs'],
                          noise_filter[0], noise_filter[1],
                          method='iir',
                          verbose=0).T

    eeg = abs(hilbert3(eeg))

    return eeg

def apply_filters(data, frequency_bands):
    # Handlers for applying different filters 
    # and create different feature bands
    frequency_bands = frequency_bands.copy()

    for k, v in frequency_bands.items():
        frequency_bands[k] = \
            {"band": v,
             "data": filter_eeg(data,
                                band=v,
                                line_freq=LINE_NOISE)}
    # Check for failed filters
    keys_to_delete = [k for k, v in frequency_bands.items()\
                        if v['data'] is None]
    for key in keys_to_delete:
        del frequency_bands[key]

    data['features']['frequency_bands'] = frequency_bands
    return data