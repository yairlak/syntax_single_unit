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

settings = load_settings_params.Settings(args.patient)
events, event_id, metadata = data_manip.get_events(args.patient, args.level)
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
tmin, tmax = -0.6, 3
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

