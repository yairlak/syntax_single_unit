# This script generates epochs from raw data. The epochs can be generated for different:
# 1. Data type: micro/macro/spikes
# 2. Temporal levels: sentence-onset/sentence-offset/word/phoneme
# 3. Filtering: raw/high-gamma/gaussian-smoothed
# 
# The script saves the epoch object to Data/UCLA/patient_?/Epochs/

import os, argparse, re, sys, glob
# Set current working directory to that of script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
from utils import load_settings_params, read_logs_and_features, convert_to_mne, data_manip, analyses
import mne
from mne.io import _merge_info
import numpy as np
from pprint import pprint
import math
from scipy import signal
from scipy import stats
from sklearn.preprocessing import RobustScaler

parser = argparse.ArgumentParser()
parser.add_argument('--patient', default='502', help='Patient number')
parser.add_argument('--data-type', default='micro', help='macro/micro/spike')
parser.add_argument('--filter', default='gaussian-kernel', help='raw/high-gamma/gaussian-kernel')
parser.add_argument('--level', default='sentence_onset', choices=['sentence_onset', 'sentence_offset', 'word', 'phone'], help='sentence_onset/sentence_offset/word/phone level')
parser.add_argument('--dont-overwrite', default=False, action='store_true')
args = parser.parse_args()
args.patient = 'patient_' + args.patient
print(args)

if args.data_type == 'spike' and args.filter=='high-gamma':
    raise('no need to epoch spike with high-gamma filtering')

blocks = range(1,7)
sfreq = 1000 # All data types (micro/macro/spike) are downsamplled to 1000Hz by generate_mne_raw.py

#TODO: add log to power

print('Loading settings, params and preferences...')
settings = load_settings_params.Settings(args.patient)
params = load_settings_params.Params(args.patient)
preferences = load_settings_params.Preferences()
pprint(preferences.__dict__); pprint(settings.__dict__); pprint(params.__dict__)

print('Logs: Reading experiment log files from experiment...')
log_all_blocks = {}
for block in blocks:
    log = read_logs_and_features.read_log(block, settings)
    log_all_blocks[block] = log
print('Preparing meta-data')
metadata = read_logs_and_features.prepare_metadata(log_all_blocks, settings, params)


##########
# EVENTS #
##########

if args.level == 'sentence_onset':
    metadata = metadata.loc[((metadata['block'].isin([1,3,5]))&(metadata['word_position']==1)) | ((metadata['block'].isin([2,4,6]))&(metadata['word_position']==1)&(metadata['phone_position'] == 1))] 
    tmin, tmax = (-1, 3.5)
elif args.level == 'sentence_offset':
    metadata = metadata.loc[(metadata['word_position'] == 0)] # filter metadata to only sentence-offset events
    tmin, tmax = (-3.5, 1.5)
elif args.level == 'word':
    metadata = metadata.loc[((metadata['first_phone'] == 1) & (metadata['block'].isin([2,4,6]))) | ((metadata['block'].isin([1,3,5])) & (metadata['word_position']>0))] # filter metadata to only word-onset events (first-phone==-1 (visual blocks))
    tmin, tmax = (-0.6, 1.5)
elif args.level == 'phone':
    metadata = metadata.loc[(metadata['block'].isin([2,4,6])) & (metadata['phone_position']>0)] # filter metadata to only phone-onset events in auditory blocks
    tmin, tmax = (-0.3, 1.2)
else:
    raise('Unknown level type (sentence_onset/sentence_offset/word/phone)')
metadata.sort_values(by='event_time')

# First column of events object
times_in_sec = sorted(metadata['event_time'].values)
min_diff_sec = np.min(np.diff(times_in_sec))
print(min_diff_sec)
print("min diff in msec: %1.2f" % (min_diff_sec * 1000))
curr_times = sfreq * metadata['event_time'].values # convert from sec to samples.
curr_times = np.expand_dims(curr_times, axis=1)

# Second column
second_column = np.zeros((len(curr_times), 1))

# Third column
event_numbers = 100 * metadata['block'].values  # For each block, the event_ids are ordered within a range of 100 numbers block1: 101-201, block2: 201-300, etc.
event_type_names = ['block_' + str(i) for i in metadata['block'].values]
event_numbers = np.expand_dims(event_numbers, axis=1)

# EVENT object: concatenate all three columns together (then change to int and sort)
events = np.hstack((curr_times, second_column, event_numbers))
events = events.astype(int)
sort_IX = np.argsort(events[:, 0], axis=0)
events = events[sort_IX, :]
# EVENT_ID dictionary: mapping block names to event numbers
event_id = dict([(event_type_name, event_number[0]) for event_type_name, event_number in zip(event_type_names, event_numbers)])

###################
# Load RAW object #
###################
fname_raw = '%s_%s_%s-raw.fif' % (args.patient, args.data_type, args.filter)
raw = mne.io.read_raw_fif(os.path.join(settings.path2rawdata, fname_raw), preload=True)
print(raw)
print(raw.ch_names)

############
# EPOCHING #
############
# First epoch then filter if needed
print(raw.first_samp)
print(events)
print(tmin, tmax)
#reject = {'seeg':4e3}
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, metadata=metadata, baseline=None, preload=True, reject=None)
print(epochs)
#print(np.sum(epochs._data, axis=2))
if any(epochs.drop_log):
    print('Dropped:')
    print(epochs.drop_log)


############################
# Robust Scaling Transform #
############################
data = epochs.copy().get_data()
for ch in range(data.shape[1]):
    transformer = RobustScaler().fit(np.transpose(data[:,ch,:]))
    epochs._data[:,ch,:] = np.transpose(transformer.transform(np.transpose(data[:,ch,:])))


########
# SAVE #
########
if not os.path.exists(settings.path2epoch_data):
    os.makedirs(settings.path2epoch_data)

fname = '%s_%s_%s_%s-epo.fif' % (args.patient, args.data_type, args.filter, args.level)
epochs.save(os.path.join(settings.path2epoch_data, fname), split_size='1.8GB', overwrite=(not args.dont_overwrite))
print('epochs saved to: %s' % os.path.join(settings.path2epoch_data, fname))




#elif args.filter == 'high-gamma':
    # -------------------------------
    # - delta_F =  2 * F / n_cycles -
    # - delta_T = n_cycles / F / pi -
    # - delta_T * delta_F = 2 / pi  -
    # -------------------------------
#    freqs = np.arange(70, 150, 15)
#    n_cycles = freqs / 6. # -> delta_T = 1/6/pi ~= 50ms; delta_F = 12Hz.
#    #n_cycles = 10
#    power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_jobs=1, average=False, n_cycles=n_cycles, return_itc=False)
#    print(power.data.shape)
#    #power.apply_baseline(baseline=(None, 0), mode='logratio')
#    epochs = mne.EpochsArray(np.average(power.data, axis=2), power.info, tmin=np.min(power.times), metadata=power.metadata, events=power.events, event_id=power.event_id)
#    if any(epochs.drop_log):
#        print('Dropped:')
#        print(epochs.drop_log)

