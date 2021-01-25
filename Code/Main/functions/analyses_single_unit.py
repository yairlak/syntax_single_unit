from __future__ import division
import numpy as np
import os
import mne
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter
from functions import data_manip, convert_to_mne
from functions.auxilary_functions import  smooth_with_gaussian

def generate_epochs_spikes(channel_num, channel_name, events_spikes, event_id, metadata, settings, params, preferences):

    print('Loading h5 file for CSC%i'%channel_num)
    spikes, channel_name = data_manip.load_combinato_sorted_h5(channel_num, channel_name, settings)
    #print(spikes)

    if len(spikes) > 0:
        print('Generating MNE raw object for spikes...')
        raw_spikes = convert_to_mne.generate_mne_raw_object_for_spikes(spikes, channel_name, settings, params)

        print('Epoching spiking data...')
        epochs_spikes = mne.Epochs(raw_spikes, events_spikes, event_id, params.tmin, params.tmax, metadata=metadata,
                                   baseline=None, preload=True, picks=None) 
        print(epochs_spikes)

        print('Generating EpochsTFR for spikes...')
        sfreq = epochs_spikes.info['sfreq']
        gaussian_width = 20 * 1e-3
        data = []
        for cluster in range(epochs_spikes._data.shape[1]):
            data_cluster = []
            for trial in range(epochs_spikes._data.shape[0]):
                mean_spike_count = epochs_spikes._data[trial, cluster, :]
                smoothed_trial_curr_cluster = smooth_with_gaussian(mean_spike_count, sfreq, gaussian_width=gaussian_width * sfreq)  # smooth with 20ms gaussian
                data_cluster.append(smoothed_trial_curr_cluster)
            data.append(np.asarray(data_cluster))
        data = np.asarray(data)
        data = np.swapaxes(data, 0, 1)
        data = np.expand_dims(data, 2)
        freqs = [1]
        epochsTFR_spikes = mne.time_frequency.EpochsTFR(epochs_spikes.info, data, epochs_spikes.times, freqs, events=epochs_spikes.events, event_id=epochs_spikes.event_id, metadata=epochs_spikes.metadata)
    else:
        epochs_spikes = []
        epochsTFR_spikes = []

    return epochs_spikes, epochsTFR_spikes

