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
parser.add_argument('--patient', default='491', type=str)
args = parser.parse_args()

# LOAD MICROPHONE DATA FROM MAT FILE
path2mic = os.path.join(f'../../../Data/UCLA/patient_{args.patient}',
                        'Raw', 'microphone')


fn_wav = os.path.join(path2mic, 'MICROPHONE_clean.wav')
rate, data_mic = sio.wavfile.read(fn_wav)

fn_mat = os.path.join(path2mic, 'MICROPHONE.mat')
sio.savemat(fn_mat, {'data': data_mic, 'samplingInterval': 1e3/rate})
