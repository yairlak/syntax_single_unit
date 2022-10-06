#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:02:02 2022

@author: yair
"""

import os
import glob
import pickle
import argparse
import pandas as pd
from decoding.utils import get_args2fname
from utils.utils import dict2filename

parser = argparse.ArgumentParser()
parser.add_argument('--smooth', default=50, type=int,
                    help='gaussian width in [msec]')
parser.add_argument('--decimate', default=50, type=int)
parser.add_argument('--side', default=8, type=float, help='Side of cube in mm')

# QUERY
parser.add_argument('--comparison-name', default='dec_quest_len2',
                    help='See Utils/comparisons.py')
parser.add_argument('--comparison-name-test', default=None,
                    help='See Utils/comparisons.py')
parser.add_argument('--block-train', choices=['auditory', 'visual'],
                    default='auditory',
                    help='Block type is added to the query in the comparison')
parser.add_argument('--block-test', choices=['auditory', 'visual'],
                    default=None,
                    help='Block type is added to the query in the comparison')
parser.add_argument('--path2output', default='../../Output/decoding')
args = parser.parse_args()


args2fname = ['comparison_name', 'comparison_name_test',
              'block_train', 'block_test',
              'smooth', 'decimate',
              'side']   # List of args
fn_pattern = dict2filename(args.__dict__, '_', args2fname, '', True)
fns = glob.glob(os.path.join(args.path2output, fn_pattern + '*.pkl'))


results = []
for fn in fns:
    results.append(pickle.load(open(fn, 'rb')))
    
df = pd.DataFrame(results, columns=['scores', 'pvals', 'times',
                                    'temp_estimator', 'clf', 'comparisons',
                                    'stimuli', 'args'])

df.to_json(os.path.join(args.path2output, 'df_' + fn_pattern + '.json'))
