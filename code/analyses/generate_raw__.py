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
import scipy
from utils.preprocessing import laplacian_reference
from utils.preprocessing import clip_data_with_RobustScaler
from utils.preprocessing import extract_spectral_activity_with_pca
import neo

print(neo.__version__)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser()
parser.add_argument('--patient', default='530', help='Patient number')
parser.add_argument('--data-type',
                    choices=['micro', 'macro', 'spike', 'microphone'],
                    default='macro', help='macro/micro/spike')
parser.add_argument('--filter', default='high-gamma',
                    choices=['raw', 'beta', 'high-gamma'])
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



##############
# DOWNSAMPLE #
##############
if args.data_type != 'microphone' and raw.info['sfreq'] > args.sfreq_downsample:
    print('Resampling data %1.2f -> %1.2f' % (raw.info['sfreq'], args.sfreq_downsample))
    raw = raw.resample(args.sfreq_downsample, npad='auto')

if args.data_type not in ['spike', 'microphone']:
    #############
    # CLIP DATA #
    #############
    print("Clipping based on robust scaling")
    raw._data = clip_data_with_RobustScaler(raw.copy().get_data(),
                                            lower=-5, upper=5,
                                            scale=False)
    
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
    
    #############################################
    # SPECTRAL WITH ZSCORE AND PCA ACROSS BANDS #
    #############################################    
    
    if args.filter=='high-gamma':
        print('Computing high-gamma')
        bands_centers = [73.0, 79.5, 87.8, 96.9, 107.0, 118.1, 130.4, 144.0] # see e.g., Moses, Mesgarani..Chang, (2016)
        bands_low = [70, 76, 83, 92.6, 101.2, 112.8, 123.4, 137.4]
        bands_high = [76, 83, 92.6, 101.2, 112.8, 123.4, 137.4, 150.6]
        raw = extract_spectral_activity_with_pca(raw, 
                                               bands_centers,
                                               bands_low, bands_high)
    elif args.filter=='beta':
        print('Computing beta activity')
        bands_centers = [15, 21, 27]
        bands_low = [12, 18, 24]
        bands_high = [18, 24, 30]
        raw = extract_spectral_activity_with_pca(raw, 
                                               bands_centers,
                                               bands_low, bands_high)
         
    ##############
    # ZSCORE RAW #
    ##############
    elif args.filter=='raw':
        raw._data = scipy.stats.zscore(raw.copy().get_data(), axis=1)

filename = '%s_%s_%s-raw.fif' % (args.patient, args.data_type, args.filter)
os.makedirs(os.path.join(path2rawdata, 'mne'), exist_ok=True)
raw.save(os.path.join(path2rawdata, 'mne', filename), overwrite=True)
print('Raw fif saved to: %s' % os.path.join(path2rawdata, filename))

