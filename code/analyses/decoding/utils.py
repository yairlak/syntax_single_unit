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

    return args


def get_args2fname(args):

    list_args2fname = ['comparison_name', 'block_train']
    if args.comparison_name_test:
        list_args2fname += ['comparison_name_test']
    if args.block_test:
        list_args2fname += ['block_test']
    if args.data_type_filters:
        list_args2fname += ['data_type_filters']
    else:
        list_args2fname += ['data_type', 'filter']
    if args.ROIs:
        list_args2fname.append('ROIs')
    elif args.probe_name:
        list_args2fname.append('probe_name')
    list_args2fname += ['smooth', 'decimate']
    if args.responsive_channels_only:
        list_args2fname += ['responsive_channels_only']

    return list_args2fname



