#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 23:31:11 2021

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
import librosa  

parser = argparse.ArgumentParser()
parser.add_argument('--patient', default = '489')
args = parser.parse_args()

settings = load_settings_params.Settings('patient_' + args.patient)
path2stimuli = '../../../Paradigm/Stimuli/Audio/'
path2mic = os.path.join(settings.path2patient_folder, 'Raw')


def get_wav_onset(mic_data, wav_data):
    xcorr = np.correlate(wav_data, mic_data, mode='full')
    fig, ax = plt.subplots(1)
    ax.plot(xcorr)
    t_onset = np.argmax(xcorr)
    return t_onset

# LOAD AND PLOT MIC DATA
mic = sio.loadmat(os.path.join(path2mic, 'MICROPHONE.mat'))
dx_mic = mic['samplingInterval'][0,0]/1e3
sfreq_mic = 1/dx_mic
mic_data = mic['data'][0,:]
# fig, ax = plt.subplots(1)
# ax.plot(np.arange(len(mic_data))/sfreq_mic, mic_data, color='k')
# DOWNSAMPLE MIC
# dx_down = 1e-3
# sfreq_down = 8000
# assert sfreq_mic % sfreq_down == 0 
# num_mic = mic_data.size
# num_downsampled = int(num_mic * dx_mic/dx_down)
# mic_downsampled = signal.decimate(mic_data, int(sfreq_mic/sfreq_down))
# mic_downsampled = mic_downsampled/max(abs(mic_downsampled))
# ax.plot(np.arange(len(mic_downsampled))*dx_down, mic_downsampled, color='r')

for i_wav in range(1, 153):
    fn_wav = os.path.join(path2stimuli, f'{i_wav}.wav')
    wav_downsampled, sfreq_down = librosa.load(fn_wav, sr=sfreq_mic) # Downsample 44.1kHz to 8kHz
    # wav_downsampled = wav_downsampled/max(abs(wav_downsampled))
    t = get_wav_onset(mic_data, wav_downsampled)
    print(t)