import argparse, os, re, sys, math
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
from utils import load_settings_params, read_logs_and_features, convert_to_mne, data_manip, analyses
from mne.io import _merge_info
import numpy as np
from pprint import pprint
import pandas as pd
from utils.read_logs_and_features import extend_metadata
from utils.data_manip import get_events

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

print('-'*100)
_, _, metadata_phone = get_events(patient, 'phone', 'micro')
_, _, metadata_word = get_events(patient, 'word', 'micro')

metadata_audio = extend_metadata(metadata_phone)
metadata_visual = metadata_word.query('block in [1, 3, 5]')
metadata_visual = extend_metadata(metadata_visual)

metadata = pd.concat([metadata_audio, metadata_visual], axis=0)
metadata = metadata.sort_values(by='event_time')

print(list(metadata))

#metadata = read_logs_and_features.prepare_metadata(log_all_blocks, settings, params, 'micro')
#metadata = read_logs_and_features.extend_metadata(metadata)
for k in sorted(list(metadata)):
    if k == 'semantic_features': print(k, ' : not showing (too many values)'); continue
    k_values = list(set(metadata[k].values))
    if any([v!=v for v in k_values]):
        print(k, len(k), 'contains nan values')
    else:
        if len(k_values) < 10:
            print(k, ':', k_values)
        else:
            print(k,  '(too many values, showing first 10):', k_values[:10])
            #print(k, len(k), k_values[:20])
        # Add also word_startswith
    #    if k == 'word_string':
    #        word_strings = list(set(metadata[k].values))
    #        word_startswith = list(set([w[0] for w in word_strings]))
    #        print(word_startswith)
print('-'*100)
print('num samples:', len(metadata[k]))
print('-'*100)
print(list(metadata))
print('-'*100)

#for index, row in metadata.iterrows():
#    print(row['sentence_string'], row['pos'])
