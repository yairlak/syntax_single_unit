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
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--smooth', default=50, type=int,
                    help='gaussian width in [msec]')
parser.add_argument('--decimate', default=50, type=int)
parser.add_argument('--side-half', default=5.0, type=float, help='Side of cube in mm')
parser.add_argument('--stride', default=5.0, type=float, help='Stride of searchlight')
# QUERY
parser.add_argument('--comparison-name', default='dec_quest_len2_end',# number_subject number_verb unacc_unerg_dec embedding_vs_long_end,
                    help='See Utils/comparisons.py')
parser.add_argument('--comparison-name-test', default='dec_quest_len2_end', # number_subject number_verb unacc_unerg_dec dec_quest_len2_end
                    help='See Utils/comparisons.py')
parser.add_argument('--block-train', choices=['auditory', 'visual'],
                    default='auditory',
                    help='Block type is added to the query in the comparison')
parser.add_argument('--block-test', choices=['auditory', 'visual'],
                    default='visual',
                    help='Block type is added to the query in the comparison')
parser.add_argument('--path2output', default='../../Output/decoding')
args = parser.parse_args()

args2fname = get_args2fname(args)
#args2fname = ['comparison_name', 'block_train']
args2fname = ['comparison_name']

# if args.block_test != args.block_train:
if args.comparison_name_test != args.comparison_name:
    args2fname.append('comparison_name_test')
    fn_pattern = f'{args.comparison_name}_*_{args.comparison_test}_*_{args.smooth}_{args.decimate}_{args.side_half}_{args.stride}'   # List of args
else:
    fn_pattern = f'{args.comparison_name}_*_{args.smooth}_{args.decimate}_{args.side_half}_{args.stride}'   # List of args
    #args2fname.append('block_test')

# fn_pattern = dict2filename(args.__dict__, '_', args2fname, '', True)
print(f'Collecting: {fn_pattern}' + '*.pkl')
#fn_pattern = 'embedding_vs_long_auditory_macro'
#fn_pattern = 'number_all_auditory_number_all_visual_macro'
# fn_pattern = 'embedding_vs_long_auditory_embedding_vs_long_visual'
# fn_pattern = 'embedding_vs_long_visual_macro'
fns = glob.glob(os.path.join(args.path2output, fn_pattern + '*.pkl'))
print(f'Found {len(fns)} files.')

results = []
for fn in tqdm(fns):
    results.append(pickle.load(open(fn, 'rb')))
    
df = pd.DataFrame(results, columns=['scores', 'pvals', 'times',
                                    'temp_estimator', 'clf', 'comparisons',
                                    'stimuli', 'args'])
assert not df.empty

df['block_train'] = df.apply(lambda row: row['args'].block_train, axis=1)
df['block_test'] = df.apply(lambda row: row['args'].block_test, axis=1)

fn_out = os.path.join(args.path2output, 'df_' + fn_pattern + '.json')
df.to_json(fn_out)
print(df)
print(f'DataFrame with results saved to: {fn_out}')
