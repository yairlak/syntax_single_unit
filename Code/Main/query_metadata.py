import argparse, os, re, sys, math
from functions import load_settings_params, read_logs_and_features, convert_to_mne, data_manip, analyses
from mne.io import _merge_info
import numpy as np
from pprint import pprint
import pandas


parser = argparse.ArgumentParser()
parser.add_argument('--query', '-q', type=str, default='')
parser.add_argument('--columns', '-c', nargs="*", default='')
parser.add_argument('--print-all', action='store_true', default=False)
parser.add_argument('--set-sentence-strings', action='store_true', default=False)
parser.add_argument('--set-word-strings', action='store_true', default=False)
args = parser.parse_args()
print(args)

# GET METADATA BY READING THE LOGS FROM THE FOLLOWING PATIENT:
patient = 'patient_479_11'
#print('Loading settings, params and preferences...')
settings = load_settings_params.Settings(patient)
params = load_settings_params.Params(patient)
preferences = load_settings_params.Preferences()

#print('Metadata: Loading features and comparisons from Excel files...')
features = read_logs_and_features.load_features(settings)

#print('Logs: Reading experiment log files from experiment...')
log_all_blocks = {}
for block in range(1, 7):
    log = read_logs_and_features.read_log(block, settings)
    log_all_blocks[block] = log

#print('Loading POS tags for all words in the lexicon')
word2pos = read_logs_and_features.load_POS_tags(settings)
#[print(k, v) for (k, v) in word2pos.items() if v in ['VB', 'VBZ']]

print('Preparing meta-data')
metadata = read_logs_and_features.prepare_metadata(log_all_blocks, settings, params)
metadata = read_logs_and_features.extend_metadata(metadata)
print(list(metadata))

if args.query:
    data_queried = metadata.query(args.query)
    if args.set_sentence_strings:
        sentence_strings = sorted(list(set(data_queried['sentence_string'].tolist())))
        [print(s) for s in sentence_strings]
        sys.exit()
    if args.set_word_strings:
        word_strings = sorted(list(set(data_queried['word_string'].tolist())))
        [print(w) for w in word_strings]
        sys.exit()
    if args.columns:
        data_queried = data_queried[args.columns]
    if args.print_all:
        pandas.set_option('display.max_rows', data_queried.shape[0]+1)
    print(data_queried)
else:
    if args.print_all:
        pandas.set_option('display.max_rows', data_queried.shape[0]+1)
    print(metadata)
