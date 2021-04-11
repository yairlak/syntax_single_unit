#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 09:01:17 2021

@author: yl254115
"""

import sys
sys.path.append('..')
from utils import load_settings_params
import os, argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage import correlate1d
import librosa  

def remove_outliers(times_log, times_device, i_log, args):
    
    # 483
    if args.patient == '483' and i_log == 0:
        times_log = times_log[3:]
        times_device = times_device[3:]
    # if args.patient == '483' and i_log == 1:
    #     times_log = times_log[1:]
    #     times_device = times_device[1:]
        
    if args.patient == '489' and i_log == 3: #something went wrong with the 4th block of patient 489; remove 3 first triggers
        times_log = times_log[3:]
        times_device = times_device[3:]
    if args.patient == '515' and i_log == 0: 
        times_log = times_log[5:]
        times_device = times_device[5:]
    
    return times_log, times_device


def refine_with_microphone(t_estimated_sec, fn_wav, args, dt=10, viz=False):
    '''
    

    Parameters
    ----------
    t_estimated_sec : TYPE
        DESCRIPTION.
    fn_wav : TYPE
        DESCRIPTION.
    args : TYPE
        DESCRIPTION.
    dt : TYPE, optional
        DESCRIPTION. The default is 2.
        half-window size in seconds [sec]

    Returns
    -------
    t_mic : TYPE
        DESCRIPTION.

    '''
    
    settings = load_settings_params.Settings('patient_' + args.patient)
    path2stimuli = '../../../Paradigm/Stimuli/Audio/'
    path2mic = os.path.join(settings.path2patient_folder, 'Raw')

    # LOAD MIC DATA
    mic = sio.loadmat(os.path.join(path2mic, 'MICROPHONE.mat'))
    dx_mic = mic['samplingInterval'][0,0]/1e3
    sfreq_mic = 1/dx_mic
    mic_data = mic['data'][0,:]
    # CROP WINDOW FROM MIC DATA
    st, ed  = int((t_estimated_sec - dt)*sfreq_mic), int((t_estimated_sec + dt)*sfreq_mic) # in sampes
    mic_data = mic_data[st:ed]
    # LOAD WAV DATA AND DOWNSAMPLE TO MIC SFREQ    
    fn_wav = os.path.join(path2stimuli, fn_wav)
    wav_downsampled, sfreq_down = librosa.load(fn_wav, sr=sfreq_mic) # Downsample 44.1kHz to 8kHz
    # CALC SPECTROGRMAS
    frequencies_mic, times_mic, spectrogram_mic = log_specgram(mic_data, sfreq_mic)
    frequencies_wav, times_wav, spectrogram_wav = log_specgram(wav_downsampled, sfreq_down)
    frequencies_mic, frequencies_wav = frequencies_mic[:160], frequencies_wav[:160]
    spectrogram_mic, spectrogram_wav = spectrogram_mic[:, :160], spectrogram_wav[:, :160]
    # CONVOLVE
    # xcorr_wavform = correlate1d(mic_data, wav_downsampled, mode='constant', cval=0) # pad with zeros (cval=0)
    xcorr_wavform = np.correlate(mic_data, wav_downsampled, mode='full')**2 # pad with zeros (cval=0)
    # CALCULATE TIME ONSET
    pos_max = np.argmax(xcorr_wavform) + 1 - (len(wav_downsampled) - 1)
    start_sample = st + pos_max
    thresh = np.mean(xcorr_wavform) + 5*np.std(xcorr_wavform)
    if np.max(xcorr_wavform) > thresh:
        t_mic = start_sample / sfreq_mic # in sec
    else:
        t_mic = t_estimated_sec # if max cross-corr is not enough distinguish then returns original estimated onset time
        print(f'Warning: max cross-corrleation is not high enough - returns original estimated time ({fn_wav})')
    
    
    # PLOT
    if viz:
        dir_figures = f'../../../Figures/microphone/patient_{args.patient}'
        os.makedirs(dir_figures, exist_ok=True)
        fig, axs = plt.subplots(3,1)
        axs[0].plot(np.arange(len(mic_data))/sfreq_mic, mic_data, color='k')
        axs[0].set_title('Microphone data')
        axs[1].plot(np.arange(len(wav_downsampled))/sfreq_down, wav_downsampled, color='k')
        axs[1].set_title('Stimulus wavform')
        axs[2].plot(xcorr_wavform)
        axs[2].axhline(thresh, color='r', ls='--')
        axs[2].set_title('Cross-correlation wavforms')
        fn_fig = f'cross_correlation_waveforms_pt_{args.patient}_{os.path.basename(fn_wav)}.png'
        fig.savefig(os.path.join(dir_figures, fn_fig))
        # SPECTROGRAM    
        # assert spectrogram_mic.shape[1] == spectrogram_wav.shape[1]
        # xcorr_spectro = []
        # for f in range(spectrogram_mic.shape[1]):
        #     # print(f'cross-correlating freq #{f}')
        #     xcorr_spectro.append(np.correlate(spectrogram_mic[:, f], spectrogram_wav[:, f], mode='full')**2) # pad with zeros (cval=0)
        # xcorr_spectro = np.asarray(xcorr_spectro).mean(axis=0)
        
        fig, axs = plt.subplots(3,1)
        axs[0].pcolormesh(times_mic, frequencies_mic, spectrogram_mic.T)
        axs[0].set_title('Microphone spectrogram')
        axs[1].pcolormesh(times_wav, frequencies_wav, spectrogram_wav.T)
        axs[1].set_title('Stimulus spectrogram')
        fn_fig = f'cross_correlation_spectrograms_pt_{args.patient}_{os.path.basename(fn_wav)}.png'
        fig.savefig(os.path.join(dir_figures, fn_fig))
        # axs[2].plot(xcorr_spectro)
        # axs[2].axhline(np.mean(xcorr_spectro) + 5*np.std(xcorr_spectro), color='r', ls='--')
        # axs[2].set_title('Cross-correlation wavforms')
    
        
    return t_mic


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)