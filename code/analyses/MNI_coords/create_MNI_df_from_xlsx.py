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

def create_bipoloar_coords_macro(df):
    
    df_macro = df.query('ch_type=="macro"')
    df_rest = df.query('ch_type!="macro"')
    
    list_dicts = []
    probe_names = df_macro['electrode'].to_list()
    probe_names = list(set(['-'.join(p.split('-')[:-1]) for p in probe_names]))
    for probe_name in probe_names:
        df_temp = df_macro.query(f'probe_name == "{probe_name}"')
        nums = sorted(df_temp['ch_num'].to_list())
        for ch_num in range(min(nums), max(nums)):
            df_bipolar = df_temp.query(f'ch_num=={ch_num} | ch_num=={ch_num+1}')
            d = {}
            d['patient'] = df_bipolar['patient'].values[0]
            d['electrode'] = f"{df_bipolar['probe_name'].values[0]}{df_bipolar['ch_num'].values[0]}-{df_bipolar['probe_name'].values[1]}{df_bipolar['ch_num'].values[1]}"
            d['probe_name'] = probe_name
            d['ch_num'] = f"{df_bipolar['ch_num'].values[0]}-{df_bipolar['ch_num'].values[1]}"
            d['ch_type'] = 'macro'
            d['MNI_x'] = df_bipolar['MNI_x'].mean()
            d['MNI_y'] = df_bipolar['MNI_y'].mean()
            d['MNI_z'] = df_bipolar['MNI_z'].mean()
            list_dicts.append(d)
    
    df_macro_new = pd.DataFrame(list_dicts)
    
    return pd.concat([df_macro_new, df_rest])

dfs = []
for fn in fns:
    print(fn)
    # PATIENT NUMBER FROM XLS FILENAME
    patient = os.path.basename(fn)
    patient = patient[4:7]
    # LOAD XLSX
    df = pd.read_excel(fn)
    # ADD COLUMNS TO DATAFRAME
    df['patient'] = patient
    df['ch_type'] = df.apply(lambda row: get_channel_type(row), axis=1)
    
    
    probe_names, ch_nums = [], []
    for ch_name, ch_type  in zip(df['electrode'], df['ch_type']):
        ch_num = int(ch_name.split('-')[-1])
        if ch_type=='macro':
            #probe_name = ''.join([c for c in ch_name if not c.isdigit()])
            probe_name = '-'.join(ch_name.split('-')[:-1])
            if probe_name[-1] == '-': probe_name = probe_name[:-1]
        if ch_type=='micro':
            probe_name = ch_name.split('micro')[0][:-1]
        if ch_type=='stim':
            probe_name = ch_name.split('stim')[0][:-1]
        probe_names.append(probe_name)
        ch_nums.append(ch_num)
    df['probe_name'] = probe_names
    df['ch_num'] = ch_nums
    
    df = create_bipoloar_coords_macro(df)
    
    
    # APPEND
    dfs.append(df)
    
df = pd.concat(dfs)
df = df.sort_values('patient')

fn_out = 'electrode_locations.csv'
df.to_csv(os.path.join(path2data, fn_out))