#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 17:02:01 2021

@author: yl254115
"""

import argparse
import os
import scipy.io as sio

parser = argparse.ArgumentParser()
parser.add_argument('--patient', default='499', type=str)
args = parser.parse_args()

# LOAD MICROPHONE DATA FROM MAT FILE
path2mic = os.path.join(f'../../../Data/UCLA/patient_{args.patient}',
                        'Raw', 'microphone')
fn_mat = os.path.join(path2mic, 'MICROPHONE.mat')
mic = sio.loadmat(fn_mat)

# GET DATA AND SAMPLING RATE
data_mic = mic['data'][0, :]
rate = int(1e3/mic['samplingInterval'][0, 0])

# SAVE AS A WAV FILE
fn_wav = os.path.join(path2mic, 'MICROPHONE.wav')
sio.wavfile.write(fn_wav, rate, data_mic)
