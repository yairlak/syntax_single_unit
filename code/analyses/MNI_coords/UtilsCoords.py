#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 20:52:41 2022

@author: yair
"""
import pandas as pd

def pick_channels_by_cube(df_coords, center, side,
                          isMacro=True, 
                          isMicro=False,
                          isStim=False):
    '''
    

    Parameters
    ----------
    df_coords : pandas dataframe
        with electrode names, coordinates, etc
    center : tuple
        with 3 elements for x, y, z in MNI.
    side : int
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
    
    assert isMacro or isMicro or isStim
    if not isMacro:
        df_coords = df_coords.query('ch_type!="macro"')
    
    if not isMicro:
        df_coords = df_coords.query('ch_type!="micro"')
    
    if not isStim:
        df_coords = df_coords.query('ch_type!="stim"')
    
    x_min, x_max = center[0] - side, center[0] + side
    y_min, y_max = center[1] - side, center[1] + side
    z_min, z_max = center[2] - side, center[2] + side
    query = f'(MNI_x<{x_max} & MNI_x>{x_min}) & (MNI_y<{y_max} & MNI_y>{y_min}) & (MNI_z<{z_max} & MNI_z>{z_min})'
    df_cube = df_coords.query(query)
    
    # DUPLICATE FOR 479 554
    df_others = df_cube.query('patient != 479 & patient != 554')
    df_479 = df_cube.query('patient == 479')
    if not df_479.empty:
        df_479_11 = df_479.copy()
        df_479_11['patient'] = '479_11'
        # df_479_25 = df_479.copy()
        # df_479_25['patient'] = '479_25'
        # df_others = pd.concat([df_others, df_479_11, df_479_25])
        df_others = pd.concat([df_others, df_479_11])
     
    df_554 = df_cube.query('patient == 554')
    if not df_554.empty:
        df_554_4 = df_554.copy()
        df_554_4['patient'] = '554_4'
        df_554_13 = df_554.copy()
        df_554_13['patient'] = '554_13'
        df_others = pd.concat([df_others, df_554_4, df_554_13])
        
    
    return df_others