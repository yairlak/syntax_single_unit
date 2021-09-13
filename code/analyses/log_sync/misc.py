#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 09:01:17 2021

@author: yl254115
"""
from skimage.feature import match_template
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
#import librosa  
from scipy.signal import butter, lfilter, hilbert
from scipy import stats
from sklearn.preprocessing import normalize

def remove_outliers(times_log, times_device, i_log, args):
    
    # 483
    #if args.patient == '483' and i_log == 0:
    #    times_log = times_log[3:]
    #    times_device = times_device[3:]
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


def find_max_cross_corr_microphone(mic_data, wav_data, sfreq):
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
    
    # CROSS-CORRELATE
    xcorr_wavform = match_template(np.expand_dims(mic_data, 0), np.expand_dims(wav_data, 0))
    ij = np.unravel_index(np.argmax(xcorr_wavform), xcorr_wavform.shape)
    IX_max, y = ij[::-1]
    t_mic_waveform = IX_max/sfreq
    
    # SPECTRO
    bands = [(i-150, i) for i in range(550, 2100, 150)]
    
    mic_envelopes,  wav_envelopes= [], []
    for low_cut, high_cut in bands:
        b, a = butter_bandpass(low_cut, high_cut, sfreq)
        # MICROPHONE
        mic_filt = lfilter(b, a, mic_data)
        mic_filt_periodic = np.concatenate((mic_filt[::-1], mic_filt, mic_filt[::-1]))
        mic_envelope = hilbert(mic_filt_periodic)[len(mic_filt):2*len(mic_filt)]
        mic_envelope = np.abs(mic_envelope)
        if np.isnan(mic_envelope).any():
            print('nan values in mic data after filtering')
        mic_envelopes.append(mic_envelope)
        
    # STIMULUS
        wav_filt = lfilter(b, a, wav_data)
        wav_filt_periodic = np.concatenate((wav_filt[::-1], wav_filt))
        wav_envelope = hilbert(wav_filt_periodic)[len(wav_filt):]
        wav_envelope = np.abs(wav_envelope)
        if np.isnan(wav_envelopes).any():
            print('nan values in wav data after filtering')
        wav_envelopes.append(wav_envelope)
    # CROSS-CORRELATE
    mic_envelopes, wav_envelopes = np.asarray(mic_envelopes), np.asarray(wav_envelopes)
    xcorr_filt = match_template(mic_envelopes, wav_envelopes)
    ij = np.unravel_index(np.argmax(xcorr_filt), xcorr_filt.shape)
    IX_max, y = ij[::-1]
    t_mic_spect = IX_max/sfreq # in sec
    
    
    # xcorr_both = match_template(np.concatenate((mic_envelopes, np.expand_dims(mic_data, 0)), axis=0), 
    #                             np.concatenate((wav_envelopes, np.expand_dims(wav_data, 0)), axis=0))
    xcorr1 = xcorr_wavform
    xcorr1[xcorr1<0] = 0
    xcorr2 = xcorr_filt
    xcorr2[xcorr2<0] = 0
    xcorr_both = np.multiply(xcorr1, xcorr2)
    ij = np.unravel_index(np.argmax(xcorr_both), xcorr_both.shape)
    IX_max, y = ij[::-1]
    t_both = IX_max/sfreq # in sec
    
    return t_mic_waveform, t_mic_spect, t_both, xcorr_wavform, xcorr_filt, xcorr_both


def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def load_microphone_data(t_estimated_sec, args, dt=2):
    settings = load_settings_params.Settings('patient_' + args.patient)
    path2mic = os.path.join('..', settings.path2patient_folder, 'Raw', 'microphone')

    # LOAD MIC DATA
    mic = sio.loadmat(os.path.join(path2mic, 'MICROPHONE.mat'))
    dx_mic = mic['samplingInterval'][0,0]/1e3
    sfreq_mic = 1/dx_mic
    mic_data = mic['data'][0,:]
    # CROP A WINDOW BASED ON ESTIMATED TIME FROM REGRESSION
    st, ed  = int((t_estimated_sec - dt)*sfreq_mic), int((t_estimated_sec + dt)*sfreq_mic) # in samples
    mic_data = mic_data[st:ed]
    mic_data = mic_data/np.abs(np.max(mic_data))
    mic_data -= mic_data.mean()
    
    return mic_data, sfreq_mic, st


def load_auditory_stimulus(fn_wav, target_sfreq, args):
    # settings = load_settings_params.Settings('patient_' + args.patient)
    path2stimuli = '../../../Paradigm/Stimuli/Audio/'
    fn_wav = os.path.join(path2stimuli, fn_wav)
    # Downsample 44.1kHz to microphone frequency
    wav_downsampled, sfreq_down = librosa.load(fn_wav, sr=target_sfreq)
    # scale
    wav_downsampled = wav_downsampled/np.abs(np.max(wav_downsampled))
    wav_downsampled -= wav_downsampled.mean()
    return wav_downsampled, sfreq_down


def plot_cross_correlation(mic_data, wav_downsampled, sfreq, xcorr_wavform, xcorr_filt, xcorr_both, first_sample_in_window, args, dt=2):    
    t_st_window = first_sample_in_window/sfreq
    settings = load_settings_params.Settings('patient_' + args.patient)
    dir_figures = os.path.join(settings.path2figures, 'log_sync', f'patient_{args.patient}')
    os.makedirs(dir_figures, exist_ok=True)
    fig_waveforms, axs = plt.subplots(3,2, figsize=(30, 20))
    axs[0, 0].plot(t_st_window+np.arange(len(mic_data))/sfreq, mic_data, color='k')
    axs[0, 0].axvline(t_st_window+dt, color='r', ls='--', lw=3)
    axs[2, 0].axvline(dt-1, color='r', ls='--', lw=3, ymax=0.1)
    axs[0, 1].axvline(t_st_window+dt, color='r', ls='--', lw=3)
    axs[2, 1].axvline(dt-1, color='r', ls='--', lw=3, ymax=0.1)
    
    axs[0, 0].set_title('Microphone data', fontsize=26)
    axs[0, 0].set_ylabel('Microphone', fontsize=26)
    axs[1, 0].plot(np.arange(len(wav_downsampled))/sfreq, wav_downsampled, color='k')
    axs[1, 0].plot(np.arange(len(mic_data))/sfreq, np.zeros(len(mic_data)), color='k')
    axs[1, 0].set_title('Stimulus waveform', fontsize=26)
    axs[1, 0].set_ylabel('Stimulus wav file', fontsize=26)
    # axs[2, 0].plot(np.arange(-len(wav_downsampled)+1, len(mic_data))/sfreq, stats.zscore(xcorr_wavform), 'k')
    axs[2, 0].plot(np.arange(xcorr_wavform.size)/sfreq, np.squeeze(xcorr_wavform), 'k')
    # thresh = np.mean(xcorr_wavform) + 5*np.std(xcorr_wavform)
    # axs[2, 0].axhline(5, color='k', ls='--')
    # ESTIMATED t
    # MICROPHONE-BASED t
    ij = np.unravel_index(np.argmax(xcorr_wavform), xcorr_wavform.shape)
    IX_max, y = ij[::-1]
    axs[0, 0].axvline(t_st_window+IX_max/sfreq+1, color='g', ls='--', lw=3)
    
    
    
    
    axs[2, 0].axvline(IX_max/sfreq, color='g', ls='--', lw=3, ymax=0.1)
    axs[0, 1].axvline(t_st_window+ IX_max/sfreq+1, color='g', ls='--', lw=3)
    axs[2, 1].axvline(IX_max/sfreq, color='g', ls='--', lw=3, ymax=0.1)
    axs[2, 0].set_title('Cross-correlation waveforms', fontsize=26)
    axs[2, 0].set_ylabel('Cross-correlation', fontsize=26)
    # SPECT
        
    # CALC SPECTROGRMAS
    frequencies_mic, times_mic, spectrogram_mic = log_specgram(mic_data, sfreq)
    frequencies_wav, times_wav, spectrogram_wav = log_specgram(wav_downsampled, sfreq)
    frequencies_mic, frequencies_wav = frequencies_mic[:160], frequencies_wav[:160]
    spectrogram_mic, spectrogram_wav = spectrogram_mic[:, :160], spectrogram_wav[:, :160]
    
    # fig_spectrograms, axs = plt.subplots(2,1, figsize=(20, 15))
    axs[0, 1].pcolormesh(t_st_window+times_mic, frequencies_mic, spectrogram_mic.T)
    axs[0, 1].set_title('Microphone spectrogram', fontsize=26)
    num_freqs, num_times = spectrogram_mic.T.shape
    v_min = spectrogram_wav.min()
    S = np.hstack((spectrogram_wav.T, v_min*np.ones((num_freqs, num_times-len(times_wav)))))
    axs[1, 1].pcolormesh(times_wav, frequencies_wav, spectrogram_wav.T)
    axs[1, 1].pcolormesh(times_mic, frequencies_wav, S)
    axs[1, 1].set_title('Stimulus spectrogram', fontsize=26)
    
    # cross-corr
    xcorr1 = xcorr_wavform
    xcorr1[xcorr1<0] = 0
    xcorr2 = xcorr_filt
    xcorr2[xcorr2<0] = 0
    xcorr_both = np.multiply(xcorr1, xcorr2)
    # print(xcorr_both.shape)
    axs[2, 1].plot(np.arange(xcorr_both.size)/sfreq, np.squeeze(xcorr_both), 'k')
    ij = np.unravel_index(np.argmax(xcorr_both), xcorr_both.shape)
    IX_max, y = ij[::-1]
    axs[0, 1].axvline(t_st_window+IX_max/sfreq+1, color='b', ls='--', lw=3)
    axs[2, 1].axvline(IX_max/sfreq, color='b', ls='--', lw=3, ymax=0.1)
    axs[0, 0].axvline(t_st_window+IX_max/sfreq+1, color='b', ls='--', lw=3)
    axs[2, 0].axvline(IX_max/sfreq, color='b', ls='--', lw=3, ymax=0.1)
    # axs[2, 1].axhline(5, color='k', ls='--')
    axs[2, 1].set_title('Cross-correlation spectrograms+waveforms', fontsize=26)
    axs[2, 1].set_xlabel('Time (within window)', fontsize=26)
    axs[2, 1].set_ylim([-0.2,1])
    axs[2, 0].set_ylim([-0.2,1])
   
    
    xt = axs[0, 0].get_xticks() 
    xt = np.append(xt,t_st_window+IX_max/sfreq+1)
    xt = np.append(xt,t_st_window+dt)
    xtl=xt.tolist()
    xtl[-1]=f"{t_st_window+IX_max/sfreq+1:1.2f}"
    xtl[-1]=f"{t_st_window+dt:1.2f}"
    axs[0, 0].set_xticks(xt)
    axs[0, 0].set_xticklabels(xtl)

    
    return fig_waveforms


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
