#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 20:52:41 2022

@author: yair
"""
import pandas as pd
import os

def get_channel_names(dict_row, data_type=None, oneliner=True):
    if data_type is None:
        data_type = dict_row['ch_type']
    patient = str(dict_row['patient'])
    if patient == '479':
        patient = '479_11'
    if patient == '554':
        patient = '554_13'
    path2fn = f'../../Data/UCLA/patient_{patient}/Raw/{data_type}/channel_numbers_to_names.txt'
    if os.path.isfile(path2fn):
        df_nums2names = pd.read_csv(path2fn,
                                    names=['ch_num', 'ch_name'],
                                    delim_whitespace=True)
    else:
        #print(f'WARNING: FILE NOT FOUND {path2fn}')
        return pd.DataFrame() # return empty dataframe
    
    # PICK
    if data_type == 'micro':
        probe_name = dict_row['probe_name']
        probe_names = [probe_name+str(i) for i in range(1, 9)]
        df = df_nums2names[df_nums2names['ch_name'].str.contains(f"{'|'.join(probe_names)}")]
        if oneliner: # put all channels names for this patient in a single row
            df_temp = df.copy()
            d = {}
            d['ch_num'] = None
            d['ch_name'] = ' '.join(df_temp['ch_name'].values)
            #print(d)
            df = pd.DataFrame(d, index=[0])
    
    elif data_type == 'spike':
        dfs = [] # loop over micro channels, and get all spikes from each
        for micro_ch_name in dict_row['ch_name'].split():
            target_ch_name = '-'.join(micro_ch_name.split('-')[1:]) # remove G?? from G??-chname (e.g., GA1-LSTG1 -> LSTG1)
            # Get only spike channels that have the micro-channel name in them
            df_nums2names = df_nums2names[df_nums2names['ch_name'].str.startswith(target_ch_name)]
            if not df_nums2names.empty:
                if oneliner: # put all channels names for this patient in a single row
                    d = {}
                    d['ch_num'] = None
                    d['ch_name'] = ' '.join(df_nums2names['ch_name'].values)
                    df = pd.DataFrame(d, index=[0])
                else:
                    df = df_nums2names
                dfs.append(df)
        if dfs:
            df = pd.concat(dfs)
        else:
            df = pd.DataFrame()
    else:
        raise('Unknown data type (micro or spike only)')
    
    return df

def add_spike_rows(df_cube):
    
    df_micro = df_cube.query('ch_type=="micro"').copy()
    
    dfs_spike = []
    for i_row, row in df_micro.iterrows():
        # Get all spike channels found on this micro channel
        df_spike_ch_names = get_channel_names(row, data_type='spike')
        if df_spike_ch_names.empty:
            continue
        df_spike = {}
        for c in ['patient', 'electrode', 'probe_name']:
            df_spike[c] = row[c]
        for coord in ['x', 'y', 'z']:
            df_spike[f'MNI_{coord}'] = row[f'MNI_{coord}']
        df_spike['ch_name'] = ' '.join(df_spike_ch_names['ch_name'].values)
        df_spike['ch_type'] = 'spike'
        df_spike = pd.DataFrame(df_spike, index=[0])
        #print(df_spike_ch_names)
        #print(df_spike)
        dfs_spike.append(df_spike)
    
    dfs_spike.append(df_cube) # add original dataframe before concatenating

    return pd.concat(dfs_spike)


def add_8_microwires(df_coords):
    # create 8 duplications of the micro coordinates,
    # referring to the relevant channel names (e.g., GB1-???1, ...,  GB2-???8)
    df_micro_all_patients = df_coords[df_coords["electrode"].str.contains("_micro-1")]
    dfs_micro = []
    for i_row, row in df_micro_all_patients.iterrows():
        df_micro = get_channel_names(row, data_type='micro')
        for coord in ['x', 'y', 'z']:
            df_micro[f'MNI_{coord}'] = row[f'MNI_{coord}']
        df_micro['patient'] = row['patient']
        df_micro['ch_type'] = row['ch_type']
        dfs_micro.append(df_micro)
    
    # remove original lines of microwires from dataframe
    df_coords = df_coords.query('ch_type!="micro"')
    # and replace with the 8 duplications per microwire
    dfs_micro.append(df_coords) # append all the rest together with the new 8 microwires
    return pd.concat(dfs_micro) # concatenate all together


def pick_channels_by_cube(df_coords, center, 
                          side_half,
                          isMacro=False, 
                          isMicro=False,
                          isSpike=False,
                          isStim=False):
    '''
    

    Parameters
    ----------
    df_coords : pandas dataframe
        with electrode names, coordinates, etc
    center : tuple
        with 3 elements for x, y, z in MNI.
    side_half : int
        size of half the side of the cube in MNI units.
    isMacro : boolean
        if True picks macro contacts
    isMicro : boolean
        if True picks micro contacts

    Returns
    -------
    channels : list
        DESCRIPTION.

    '''
    
    assert isMacro or isMicro or isStim or isSpike
    
    
    
    df_coords = add_8_microwires(df_coords)
    #if isSpike:
    #    df_coords = add_spike_rows(df_coords)
 
    if not isMacro: # remove macro channels if not chosen
        df_coords = df_coords.query('ch_type!="macro"')
    if not isMicro:  # remove micro channels if not chosen
        df_coords = df_coords.query('ch_type!="micro"')
    if not isStim: # remove stim channels if needed
        df_coords = df_coords.query('ch_type!="stim"')
    
    x_min, x_max = center[0] - side_half, center[0] + side_half
    y_min, y_max = center[1] - side_half, center[1] + side_half
    z_min, z_max = center[2] - side_half, center[2] + side_half
    query = f'(MNI_x<{x_max} & MNI_x>{x_min}) & (MNI_y<{y_max} & MNI_y>{y_min}) & (MNI_z<{z_max} & MNI_z>{z_min})'
    df_cube = df_coords.query(query)
     
    # DUPLICATE FOR 479 554 (Two sessions)
    df_all = df_cube.copy().query('patient != 479 & patient != 554')
    df_479 = df_cube.copy().query('patient == 479')
    if not df_479.empty:
        df_479_11 = df_479.copy()
        df_479_11['patient'] = '479_11'
        # REMOVE 479_25. problem with logs
        #df_479_25 = df_479.copy() 
        #df_479_25['patient'] = '479_25'
        #df_all = pd.concat([df_all, df_479_11, df_479_25])
        df_all = pd.concat([df_all, df_479_11])

    df_554 = df_cube.copy().query('patient == 554')
    if not df_554.empty:
        df_554_4 = df_554.copy()
        df_554_4['patient'] = '554_4'
        df_554_13 = df_554.copy()
        df_554_13['patient'] = '554_13'
        df_all = pd.concat([df_all, df_554_4, df_554_13])


    if isSpike: # For each micro channel, add lines for all spike channels (single units) on the microwire
        df_all = add_spike_rows(df_all)
    
    return df_all
