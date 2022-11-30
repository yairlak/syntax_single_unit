# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import glob
import os

path2data = '../../../Data/UCLA/MNI_coords/'
fns = glob.glob(os.path.join(path2data, '*.xlsx'))

def get_channel_type(row):
    if 'micro' in row['electrode']:
        ch_type = 'micro'
    elif 'stim' in row['electrode']:
        ch_type = 'stim'
    else:
        ch_type = 'macro'
    return ch_type

fn_out = 'electrode_locations.csv'
df = pd.read_csv(os.path.join(path2data, fn_out))

print(df.query('patient==551').to_string())
