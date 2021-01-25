# This script generates epochs from raw data. The epochs can be generated for different:
# 1. Data type: macro/micro/spikes data
# 2. Temporal levels: sentence/word/phoneme
# 3. Filtering: raw/high-gamma/gaussian-smoothed
# 
# The script saves the epoch object to Data/UCLA/patient_?/Epochs/

import os, argparse, re, sys, glob
import matplotlib.pyplot as plt
# Set current working directory to that of script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
#sys.path.append('..')
from functions import load_settings_params, read_logs_and_features, convert_to_mne, data_manip, analyses
import mne
from mne.io import _merge_info
import numpy as np
from pprint import pprint
import math
from scipy import signal
from scipy import stats
from functions.utils import smooth_with_gaussian


parser = argparse.ArgumentParser()
parser.add_argument('--patient', default='479_11', help='Patient number')
parser.add_argument('--data-type', default='micro', help='macro/micro/spike')
parser.add_argument('--dont-overwrite', default=False, action='store_true')
args = parser.parse_args()
args.patient = 'patient_' + args.patient
print(args)


print('Loading settings, params and preferences...')
settings = load_settings_params.Settings(args.patient)
params = load_settings_params.Params(args.patient)
preferences = load_settings_params.Preferences()
pprint(preferences.__dict__); pprint(settings.__dict__); pprint(params.__dict__)

# PATHS
if args.data_type == 'micro' or args.data_type == 'spike':
    path2CSC_mat = os.path.join(settings.path2rawdata, 'micro', 'CSC_mat')
elif args.data_type == 'macro':
    path2CSC_mat = os.path.join(settings.path2rawdata, 'macro', 'CSC_mat')


###################
# Load RAW object #
###################
fname_raw = '%s_%s-raw.fif' % (args.patient, args.data_type)
raw = mne.io.read_raw_fif(os.path.join(settings.path2rawdata, fname_raw), preload=True)
print(raw)
raw_filtered = raw.copy().notch_filter(params.line_frequency, n_jobs=1, fir_design='firwin') # notch filter


############
# PLOT PSD #
############
num_channels = len(raw.ch_names)
fmin = 2
fmax = 300
n_fft = 2048

print(num_channels)
for i, ch in enumerate(raw.ch_names):
    if ch == 'MICROPHONE': continue
    fig_psd, ax_psd = plt.subplots(figsize=(10,10))
    print(f'Channel {ch}')
    raw.plot_psd(fmin=fmin, fmax=fmax, n_fft=n_fft, n_jobs=1, ax=ax_psd, picks=[ch], color=(1, 0, 0), show=False, average=False, spatial_colors=False)
    raw_filtered.plot_psd(fmin=fmin, fmax=fmax, n_fft=n_fft, n_jobs=1, ax=ax_psd, picks=[ch], color=(0, 1, 0), show=False, average=False, spatial_colors=False)

    ########
    # SAVE #
    ########
    fname = 'PSD_%s_%s_ch_%i_%s.png' % (args.patient, args.data_type, i, ch)
    save2 = os.path.join(settings.path2figures, 'PSDs', fname)
    fig_psd.savefig(save2)
    print('Figures saved to: %s' % save2)
    plt.close(fig_psd)
