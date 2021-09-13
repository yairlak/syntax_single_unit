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
from utils.data_manip import DataHandler

parser = argparse.ArgumentParser()
parser.add_argument('--patient', default='502', help='Patient number')
parser.add_argument('--data-type',
                    choices=['micro', 'macro', 'spike', 'microphone'],
                    default='micro', help='macro/micro/spike')
args = parser.parse_args()
args.patient = 'patient_' + args.patient
print(args)

path2rawdata = os.path.join('..', '..', 'Data', 'UCLA',
                            f'{args.patient}', 'Raw')

# BANDS (see e.g., Moses, Mesgarani..Chang, 2016)
bands_centers = [73.0, 79.5, 87.8, 96.9, 107.0, 118.1, 130.4, 144.0]
bands_low = [70, 76, 83, 92.6, 101.2, 112.8, 123.4, 137.4]
bands_high = [76, 83, 92.6, 101.2, 112.8, 123.4, 137.4, 150.6]

#############
# LOAD DATA #
#############
data = DataHandler(args.patient, args.data_type, 'raw')
data.load_raw_data()

# Both neural and feature data into a single raw object
for p, (patient, data_type) in enumerate(zip(args.patient,
                                             args.data_type)):
    raw = data.raws[p]
    print(f'Extracting high-gamma for {patient} {data_type}')

    raw_eight_bands = []
    for i_band, (band_low, band_high) in enumerate(zip(bands_low, bands_high)):
        print(f'Filtering band {i_band}/{len(bands_low)}')
        # BAND-PASS, ANALYTIC SIGNAL (HILBERT)
        # LOG AND THEN Z-SCORE
        # num_channels X num_timepoints
        raw_eight_bands.append(scipy.stats.zscore(  # zscore
                               np.log(raw.copy().   # log power
                                      filter(band_low, band_high).  # band
                                      apply_hilbert(envelope=True).get_data()),
                               axis=1))
    # 8-features (bands) X num_channels X num_timepoints
    raw_eight_bands = np.asarray(raw_eight_bands)
    # print(raw_eight_bands.shape)

    # CLIP AND PCA ACROSS BANDS
    print('Clip and mean across bands')
    for i_channel in range(raw_eight_bands.shape[1]):
        # num_timepoints X 8-features (bands)
        data_curr_channel = raw_eight_bands[:, i_channel, :].transpose()
        # CLIP
        lower, upper = -5, 5  # zscore limits
        data_curr_channel[data_curr_channel > upper] = upper
        data_curr_channel[data_curr_channel < lower] = lower
        raw._data[i_channel, :] = data_curr_channel.mean(axis=1)

    filename = '%s_%s_%s-raw.fif' % (args.patient,
                                     args.data_type,
                                     'high-gamma')
    raw.save(os.path.join(path2rawdata, 'mne', filename), overwrite=True)
    print('Raw fif saved to: %s' % os.path.join(path2rawdata, filename))
