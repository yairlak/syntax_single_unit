#  Generate raw mne files (fif formant) from mat or combinato files:
# - In the case of data-type = 'macro' bi-polar referencing is applied.
# - Notch filtering of line noise is performed.
# - clipping using robustScalar transform is applied
#   by using -5 and 5 for lower/upper bounds.
# - The output is a raw mne object saved to Data/UCLA/patient_?/Raw/

import os
import argparse
from utils import data_manip
import numpy as np
from mne.filter import filter_data
from sklearn.preprocessing import RobustScaler
import scipy
from utils.preprocessing import laplacian_reference, hilbert3

# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)

parser = argparse.ArgumentParser()
parser.add_argument('--patient', default='505', help='Patient number')
parser.add_argument('--data-type',
                    choices=['micro', 'macro', 'spike', 'microphone'],
                    default='macro', help='macro/micro/spike')
parser.add_argument('--filter', default='raw',
                    choices=['raw', 'high-gamma'])
parser.add_argument('--from-mat',
                    default=False, action='store_true',
                    help='Load data from mat files.')
parser.add_argument('--ch-names-from-file',
                    default=False, action='store_true',
                    help='Load data from mat files.')
parser.add_argument('--sfreq-downsample', type=int,
                    default=1000, help='Downsampling frequency')
parser.add_argument('--line-frequency',
                    default=[50, 60], help='in Hz')
args = parser.parse_args()
args.patient = 'patient_' + args.patient
print(args)

path2rawdata = os.path.join('..', '..', 'Data', 'UCLA',
                                    f'{args.patient}', 'Raw')

raw = data_manip.generate_mne_raw(args.data_type,
                                  args.from_mat,
                                  path2rawdata,
                                  args.sfreq_downsample,
                                  args.ch_names_from_file)

print(raw.ch_names)

if args.data_type != 'microphone':
    ##############
    # DOWNSAMPLE #
    ##############
    if raw.info['sfreq'] > args.sfreq_downsample:
        print('Resampling data %1.2f -> %1.2f' % (raw.info['sfreq'], args.sfreq_downsample))
        raw = raw.resample(args.sfreq_downsample, npad='auto')

# PROCSESING MICRO/MACRO LFPs
if args.data_type not in ['spike', 'microphone']:
    ###########
    # DETREND #
    ###########
    print('Detrending')
    raw._data = scipy.signal.detrend(raw.copy().get_data(), axis=0)
    
    ###################
    # Basic FILTERING #
    ###################
    cutoff_l, cutoff_h = 0.5, None
    print('Filtering data from {} to {}'.format(cutoff_l, cutoff_h))
    raw._data = filter_data(raw.copy().get_data(), raw.info['sfreq'],
                            cutoff_l, cutoff_h, verbose=0)
    
    ###############
    # REFERENCING #
    ###############
    if args.data_type == 'macro': # Laplacian ref for macro
        print("Applying Laplacian reference")
        raw._data = laplacian_reference(raw.copy().get_data(), raw.ch_names)
        
    ################
    # NOTCH (line) #
    ################
    for line_freq in args.line_frequency:
        print(f"Notch filtering for: {line_freq}")
        raw.notch_filter(np.arange(line_freq, 5*line_freq, line_freq),
                         fir_design='firwin') # notch filter
    
    ##############
    # HIGH-GAMMA #
    ##############
    if args.filter=='high-gamma':
        print('Computing high-gamma')
        raw._data = filter_data(raw.copy().get_data(), raw.info['sfreq'],
                                  70, 150,
                                  method='iir',
                                  verbose=0)
        
        raw._data = abs(hilbert3(raw.copy().get_data()))
     
    ############
    # CLIPPING #
    ############
    print("Clipping based on robust scaling")
    transformer = RobustScaler().fit(raw.copy().get_data().T)
    data_scaled = transformer.transform(raw.copy().get_data().T).T # num_channels X num_timepoints
    lower, upper = -5, 5
    data_scaled[data_scaled>upper] = upper
    data_scaled[data_scaled<lower] = lower
    raw._data = data_scaled

filename = '%s_%s_%s--raw.fif' % (args.patient, args.data_type, args.filter)
os.makedirs(os.path.join(path2rawdata, 'mne'), exist_ok=True)
raw.save(os.path.join(path2rawdata, 'mne', filename), overwrite=True)
print('Raw fif saved to: %s' % os.path.join(path2rawdata, filename))

