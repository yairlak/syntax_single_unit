#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 21:14:31 2021

A tool for creating a file with channel names for BLACKROCK data

@author: yl254115
"""
import numpy as np

patient = 539
data_type = 'micro'
path2file = f'../../../Data/UCLA/patient_{patient}/Raw/{data_type}/channel_numbers_to_names.txt'

dict_init = {1:'A', 2:'B', 3:'C', 4:'D'}

if patient==510 and data_type=='micro':
    probe_names = ['RAH', 'RA', 'ROF', 'RPSM',
                   'LAH', 'LA', 'LOF', 'LPSM',
                   'RAI', 'LAF', 'LAI', 'LPC', 'LSM']
elif patient==504 and data_type=='micro':
    probe_names = ['LMI', 'LEC', 'LA', 'LPHG',
                   'RAH', 'REC', 'LOF', 'LIF', 'LAI', 'LAH']
elif patient==530 and data_type=='micro':
    probe_names = ['REC', 'RMH', 'ROF', 'RAC',
                   'LMH', 'LEC', 'LOF', 'LAC', 
                   'LA', 'LHGa', 'LPI-SMGa', 'LFOP']
elif patient==539 and data_type=='micro':
    probe_names = ['RMH', 'RA', 'ROF-AC', 'LMH',
                   'REC', 'LFSG', 'LEC']
with open(path2file, 'w') as f:
    cnt = 1
    for i_probe, probe_name in enumerate(probe_names):
        letter = dict_init[np.floor(i_probe/4) + 1]
        num = 1 + i_probe % 4
        for i_ch in range(1, 9):
            channel_name = f'G{letter}{num}-{probe_name}{i_ch}'
            f.write(f'{cnt} {channel_name}\n')
            cnt += 1
print(f'Saved to: {path2file}')            
