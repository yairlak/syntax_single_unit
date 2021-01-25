import os, argparse
import pandas as pd
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from pprint import pprint
from functions import data_manip, load_settings_params
from nilearn import plotting  
import numpy as np
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument('--patients', nargs='*', default=[])
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike'], default='micro', help='electrode type')
parser.add_argument('--level', choices=['sentence_onset', 'sentence_offset', 'word', 'phone'], default='sentence_onset', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'high-gamma'], default='raw', help='')
parser.add_argument('--block-type', choices=['visual', 'auditory'], default='visual')
parser.add_argument('--thresh', type=float, default=0.05, help='Threshold below which p-value indicates significance')

#
args = parser.parse_args()
args.patients = ['patient_' + p for p in args.patients]
print(args)
