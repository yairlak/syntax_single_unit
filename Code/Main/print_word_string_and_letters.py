import argparse, os, re, sys, math
from functions import load_settings_params, read_logs_and_features, convert_to_mne, data_manip, analyses
from mne.io import _merge_info
import numpy as np
from pprint import pprint
import pandas

# GET METADATA BY READING THE LOGS FROM THE FOLLOWING PATIENT:
patient = 'patient_479_11'
print('Loading settings, params and preferences...')
settings = load_settings_params.Settings(patient)
params = load_settings_params.Params(patient)
preferences = load_settings_params.Preferences()

print('Metadata: Loading features and comparisons from Excel files...')
features = read_logs_and_features.load_features(settings)

print('Logs: Reading experiment log files from experiment...')
log_all_blocks = {}
for block in range(1, 7):
    log = read_logs_and_features.read_log(block, settings)
    log_all_blocks[block] = log

print('Loading POS tags for all words in the lexicon')
word2pos = read_logs_and_features.load_POS_tags(settings)

print('-'*100)
print('Preparing meta-data')
metadata = read_logs_and_features.prepare_metadata(log_all_blocks, features, word2pos, settings, params, preferences)
print('-'*100)

# WORDS
word_strings = list(set(metadata['word_string']))
print('WORDS')
print('-'*100)
print(word_strings)
print(f'Number of words = {len(word_strings)}')

word_strings = list(set(metadata['word_string']))

# LETTERS
letters = []
[letters.extend(set(w)) for w in word_strings]
letters = sorted(list(set(letters)))
print(letters)
print('-'*100)
print(f'Number of characters = {len(letters)}')

# BIGRAMS
bigrams = []
for w in word_strings:
    bigrams.extend([w[k]+w[k+1] for k in range(len(list(w))-1)])
bigrams = sorted(list(set(bigrams)))
print(bigrams)
print('-'*100)
print(f'Number of bigrams = {len(bigrams)}')

# MORPHEMES
morphemes = list(set(metadata['morpheme']))
print('MORPHEMES')
print('-'*100)
print(morphemes)
