#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:56:07 2021

@author: yl254115
"""
import numpy as np
import sys
sys.path.append('..')


def get_comparisons(comparison_name, comparison_name_test=None):
    from utils import comparisons
    
    # MAIN COMPARISON
    comparisons = comparisons.comparison_list()
    comparison = comparisons[comparison_name].copy()
    
    # TEST COMPARISON
    if comparison_name_test:
        comparison_test = comparisons[comparison_name_test].copy()
    else:
        comparison_test = None

    return [comparison, comparison_test]



def update_args(args):
    from utils.utils import get_all_patient_numbers
    from utils.utils import get_patient_probes_of_region

    assert args.block_train

    if not args.patient:
        args.patient = get_all_patient_numbers(path2data='../../Data/UCLA')
        args.data_type *= len(args.patient)
        args.filter *= len(args.patient)
    args.patient = ['patient_' + p for p in  args.patient]

    if args.ROIs:
        # Overwrite args.patient and probe_name, based on region info
        assert args.data_type_filters
        args.patient, args.data_type, args.filter, args.probe_name = \
                get_patient_probes_of_region(args.ROIs, args.data_type_filters)

    if args.comparison_name == args.comparison_name_test:
        args.comparison_name_test = None
    
    if args.comparison_name_test:
        args.GAC = True # Generalization across conditions
    else:
        args.GAC = False

    if args.block_train == args.block_test:
        args.block_test = None
    
    if args.block_test:
        args.GAM = True # Generalization across modalities
        # For GAM, test comparison is same as train comparison
        args.comparison_name_test = args.comparison_name
    else:
        args.GAM = False

    if args.fixed_constraint and (not args.fixed_constraint_test):
        args.fixed_constraint_test = args.fixed_constraint

    return args


def get_args2fname(args):

    list_args2fname = ['comparison_name', 'block_train']
    if 'comparison_name_test' in args and args.comparison_name_test:
        list_args2fname += ['comparison_name_test']
    if args.block_test:
        list_args2fname += ['block_test']
    if 'data_type_filters' in args and args.data_type_filters:
        list_args2fname += ['data_type_filters']
    elif 'data_type' in args and 'filter' in args:
        list_args2fname += ['data_type', 'filter']
    if 'ROIs' in args and args.ROIs:
        list_args2fname.append('ROIs')
    elif 'probe_name' in args and args.probe_name:
        list_args2fname.append('probe_name')

    list_args2fname += ['smooth', 'decimate']

    if 'side_half' in args and args.side_half:
        list_args2fname += ['side_half']
    if 'coords' in args and args.coords:
        list_args2fname += ['coords']

    return list_args2fname


def get_channel_names_from_cube_center(df_coords,
                                       x_center, y_center, z_center,
                                       side,
                                       data_types):
    
    df_coords = df_coords.query(f'ch_type in {data_types}')
    x_constraint = f'(MNI_x <= {x_center + side}) & (MNI_x >= {x_center - side})'
    y_constraint = f'(MNI_y <= {y_center + side}) & (MNI_y >= {y_center - side})'
    z_constraint = f'(MNI_z <= {z_center + side}) & (MNI_z >= {z_center - side})'
    df_in_cube = df_coords.query(f'{x_constraint} & {y_constraint} & {z_constraint}')
    
    patients = df_in_cube['patient'].to_list()
    channel_names = df_in_cube['electrode'].to_list()
    probe_names = df_in_cube['probe_name'].to_list()
    channel_nums = df_in_cube['ch_num'].to_list()
    
    return patients, channel_names, probe_names, channel_nums

