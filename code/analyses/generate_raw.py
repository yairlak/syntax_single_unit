#  Generate raw mne files (fif formant) from mat or combinato files:
# - In the case of data-type = 'macro' bi-polar referencing is applied.
# - Notch filtering of line noise is performed.
# - clipping using robustScalar transform is applied
#   by using -5 and 5 for lower/upper bounds.
# - The output is a raw mne object saved to Data/UCLA/patient_?/Raw/

import os
import argparse
import re
from utils import load_settings_params, data_manip
import mne
import numpy as np
from pprint import pprint
from sklearn.preprocessing import RobustScaler
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
import scipy
from neo.io import NeuralynxIO
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--patient', default='515', help='Patient number')
parser.add_argument('--data-type',
                    choices=['micro', 'macro', 'spike', 'microphone'],
                    default='microphone', help='macro/micro/spike')
parser.add_argument('--filter', default='raw',
                    choices=['raw', 'high-gamma'])
parser.add_argument('--from-mat',
                    default=False, action='store_true',
                    help='Load data from mat files.')
parser.add_argument('--sfreq-downsample', type=int,
                    default=1000, help='Downsampling frequency')
parser.add_argument('--line-frequency',
                    default=50, help='in Hz')
args = parser.parse_args()
args.patient = 'patient_' + args.patient
print(args)


path2rawdata = os.path.join('..', '..', 'Data', 'UCLA',
                                    f'{args.patient}', 'Raw')

if args.data_type == 'microphone':
    path2data = os.path.join(path2rawdata, args.data_type)
else:
    path2data = os.path.join(path2rawdata, args.data_type, 'ncs')

if args.from_mat:
    raise('Implementation Error')
    channel_data = scipy.io.loadmat(os.path.join(path2rawdata, args.data_type, 'MICROPHONE.mat'))['data']
    sfreq = 32000
    info = mne.create_info(ch_names=['MIC'], sfreq=sfreq, ch_types=['seeg'])
    raw = mne.io.RawArray(channel_data, info)
else:
    reader = NeuralynxIO(dirname=path2data)
    time0, timeend = reader.global_t_start, reader.global_t_stop
    print(f'Start time {time0}, End time {timeend}')
    print(f'Sampling rate [Hz]: {reader._sigs_sampling_rate}')
    blks = reader.read(lazy=True)
    channels = reader.header['signal_channels']
    n_channels = len(channels)
    ch_names = [channel[0] for channel in channels]
    channel_nums = [channel[1] for channel in channels]
    print('Number of channel %i: %s'
                  % (len(ch_names), ch_names))

    channel_data = []
    for i_segment, segment in enumerate(blks[0].segments):
        print(i_segment)
        anasig = segment.analogsignals[0].load()
        channel_data.append(np.asarray(anasig))
        del anasig
    channel_data = np.vstack(channel_data).T
    del blks

    if args.data_type in ['micro', 'macro', 'microphone']:
        print('Loading %s CSC data' % args.data_type.upper())
        ch_types = ['seeg'] * n_channels
        sfreq = reader._sigs_sampling_rate
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(channel_data, info)
        del channel_data

if args.data_type != 'microphone':
    # Downsample
    if raw.info['sfreq'] > args.sfreq_downsample:
        print('Resampling data %1.2f -> %1.2f' % (raw.info['sfreq'], args.sfreq_downsample))
        raw = raw.resample(args.sfreq_downsample, npad='auto')

###############
# REFERENCING #
###############
if args.data_type == 'macro': #bipolar ref for macro
    print("Applying Bipolar reference")
    reference_ch = [ch for ch in raw.copy().info['ch_names']]
    anodes = reference_ch[0:-1]
    cathodes = reference_ch[1::]
    to_del = []
    for i, (a, c) in enumerate(zip(anodes,cathodes)):
         if [i for i in a if not i.isdigit()] != [i for i in c if not i.isdigit()]:
              to_del.append(i)
    for idx in to_del[::-1]:
         del anodes[idx]
         del cathodes[idx]
    #for a, c in zip(anodes, cathodes): print(a,c)
    raw = mne.set_bipolar_reference(raw.copy(), anodes, cathodes)
print(raw)
print(raw.ch_names)

###################
# Basic FILTERING #
###################
if args.data_type not in ['spike', 'microphone']:
    ################
    # NOTCH (line) #
    ################
    raw.notch_filter(np.arange(args.line_frequency, 5*args.line_frequency, args.line_frequency), fir_design='firwin') # notch filter
    raw.filter(0.05, None, fir_design='firwin') # High-pass filter
    if args.filter.startswith('gaussian-kernel') or args.filter == 'raw' and args.data_type != 'microphone':

        ############
        # CLIPPING #
        ############
        print("Clipping based on robust scaling")
        data = raw.copy().get_data()
        transformer = RobustScaler().fit(data.T)
        data_scaled = transformer.transform(data.T).T # num_channels X num_timepoints
        lower, upper = -5, 5
        data_scaled[data_scaled>upper] = upper
        data_scaled[data_scaled<lower] = lower
        raw._data = data_scaled
        # raw._data = transformer.inverse_transform(data_scaled.T).T # INVERSE TRANSFORM
    elif args.filter=='high-gamma':
        print('Extracting high-gamma')
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
            # LOG AND THEN Z-SCORE
            raw_band_hilb_zscore = scipy.stats.zscore(np.log(raw_band_hilb._data), axis=1) # num_channels X num_timepoints
            raw_eight_bands.append(raw_band_hilb_zscore)
        raw_eight_bands = np.asarray(raw_eight_bands) # 8-features (bands) X num_channels X num_timepoints
        print(raw_eight_bands.shape)

        # CLIP AND PCA ACROSS BANDS
        print('Clip and PCA across bands')
        for i_channel in range(raw_eight_bands.shape[1]):
            data_curr_channel = raw_eight_bands[:, i_channel, :].transpose() # num_timepoints X 8-features (bands)
            # CLIP
            lower, upper = -5, 5 # zscore limits
            data_curr_channel[data_curr_channel>upper] = upper
            data_curr_channel[data_curr_channel<lower] = lower
            raw._data[i_channel, :] = data_curr_channel.mean(axis=1)
            
            # PCA
            #pca = PCA(n_components=1)
            #raw._data[i_channel, :] = pca.fit_transform(data_curr_channel).reshape(-1) # first PC across the 8 bands in high-gamma


filename = '%s_%s_%s-raw.fif' % (args.patient, args.data_type, args.filter)
raw.save(os.path.join(path2rawdata, filename), overwrite=True)
print('Raw fif saved to: %s' % os.path.join(path2rawdata, filename))

