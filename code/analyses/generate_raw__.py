#  Generate raw mne files (fif formant) from mat or combinato files:
# - In the case of data-type = 'macro' Laplacian referencing is applied.
# - Detrending
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
from sklearn.decomposition import PCA

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

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
        bands_centers = [73.0, 79.5, 87.8, 96.9, 107.0, 118.1, 130.4, 144.0] # see e.g., Moses, Mesgarani..Chang, (2016)
        bands_low = [70, 76, 83, 92.6, 101.2, 112.8, 123.4, 137.4]
        bands_high = [76, 83, 92.6, 101.2, 112.8, 123.4, 137.4, 150.6]
        
        raw_eight_bands = []
        for i_band, (band_low, band_high) in enumerate(zip(bands_low, bands_high)):
            # BAND-PASS
            raw_band = raw.copy()
            raw_band.filter(band_low, band_high)
            # ANALYTIC SIGNAL (HILBERT)
            raw_band_hilb = raw_band.copy()
            raw_band_hilb.apply_hilbert(envelope=True)
            # Z-SCORE
            raw_band_hilb_zscore = scipy.stats.zscore(raw_band_hilb._data, axis=1) # num_channels X num_timepoints
            raw_eight_bands.append(raw_band_hilb_zscore)
        raw_eight_bands = np.asarray(raw_eight_bands) # 8-features (bands) X num_channels X num_timepoints
        print(raw_eight_bands.shape)
        
        print('Clip and PCA across bands')
        for i_channel in range(raw_eight_bands.shape[1]):
            data_curr_channel = raw_eight_bands[:, i_channel, :].transpose() # num_timepoints X 8-features (bands)
            # CLIP
            lower, upper = -5, 5 # zscore limits
            data_curr_channel[data_curr_channel>upper] = upper
            data_curr_channel[data_curr_channel<lower] = lower
            # raw._data[i_channel, :] = data_curr_channel.mean(axis=1)
            pca = PCA(n_components=1)
            print(data_curr_channel.shape)
            raw._data[i_channel, :] = pca.fit_transform(data_curr_channel).reshape(-1) # first PC across the 8 bands in high-gamma

         
    ############
    # CLIPPING #
    ############
    elif args.filter=='raw':      
        print("Clipping based on robust scaling")
        transformer = RobustScaler().fit(raw.copy().get_data().T)
        data_scaled = transformer.transform(raw.copy().get_data().T).T # num_channels X num_timepoints
        lower, upper = -5, 5
        data_scaled[data_scaled>upper] = upper
        data_scaled[data_scaled<lower] = lower
        raw._data = data_scaled

filename = '%s_%s_%s-raw.fif' % (args.patient, args.data_type, args.filter)
os.makedirs(os.path.join(path2rawdata, 'mne'), exist_ok=True)
raw.save(os.path.join(path2rawdata, 'mne', filename), overwrite=True)
print('Raw fif saved to: %s' % os.path.join(path2rawdata, filename))


