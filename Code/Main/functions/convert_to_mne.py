import numpy as np
import mne


def generate_events_array(metadata, params):
    '''

    :param metadata: (pandas dataframe) num_words X num_features; all words across all stimuli
    :param params: (object) general parameters
    :return:
    '''

    # First column of events object
    times_in_sec = sorted(metadata['event_time'].values)
    #print(times_in_sec)
    min_diff_sec = np.min(np.diff(times_in_sec))
    print(min_diff_sec)
    print("min diff in msec: %1.2f" % (min_diff_sec * 1000))
    curr_times = params.sfreq_raw * metadata['event_time'].values # convert from sec to samples.
    curr_times = np.expand_dims(curr_times, axis=1)

    # Second column
    second_column = np.zeros((len(curr_times), 1))

    # Third column
    event_numbers = 100 * metadata['block'].values  # For each block, the event_ids are ordered within a range of 100 numbers block1: 101-201, block2: 201-300, etc.
    event_type_names = ['block_' + str(i) for i in metadata['block'].values]
    event_numbers = np.expand_dims(event_numbers, axis=1)

    # EVENT object: concatenate all three columns together (then change to int and sort)
    events_micro = np.hstack((curr_times, second_column, event_numbers))
    events_micro = events_micro.astype(int)
    sort_IX = np.argsort(events_micro[:, 0], axis=0)
    events_micro = events_micro[sort_IX, :]
    curr_times = curr_times[sort_IX, 0]

    # EVENT_ID dictionary: mapping block names to event numbers
    event_id = dict([(event_type_name, event_number[0]) for event_type_name, event_number in zip(event_type_names, event_numbers)])

    # Generate another event object for single-unit data (which has a different sampling rate)
    events_spikes = np.copy(events_micro)
    #d = events_spikes[:, 0] * params.sfreq_spikes / params.sfreq_raw
    events_spikes[:, 0] = curr_times * params.sfreq_spikes / params.sfreq_raw
    events_spikes = events_spikes.astype(np.int64)

    # Generate another event object for single-unit data (which has a different sampling rate)
    events_macro = np.copy(events_micro)
    #d = events_macro[:, 0] * params.sfreq_macro / params.sfreq_raw
    events_macro[:, 0] = curr_times * params.sfreq_macro / params.sfreq_raw
    events_macro = events_macro.astype(np.int64)

    return events_micro, events_spikes, events_macro, event_id

def generate_mne_raw_object(data, settings, params):
    num_channels = data.shape[0]
    ch_types = ['seeg' for s in range(num_channels)]
    info = mne.create_info(ch_names=[settings.channel_name], sfreq=params.sfreq_raw, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    return raw

def generate_mne_raw_object_for_spikes(spikes, electrode_names, settings, params):
    time0_sec = settings.time0 / 1e6
    sfreq = params.sfreq_spikes
    num_groups = len(spikes)
    ch_types = ['seeg' for _ in range(num_groups)]

    info = mne.create_info(ch_names=electrode_names, sfreq=sfreq, ch_types=ch_types)

    num_samples = 1+int(sfreq * (settings.timeend - settings.time0)/1e6) # Use same sampling rate as for macro, just for convenience.
    spikes_matrix_all_groups = np.empty((0, num_samples))
    for groups, curr_spike_times_msec in enumerate(spikes):
        spikes_zero_one_vec = np.zeros(num_samples) # convert to samples from sec
        curr_spike_times_sec = [t/1e3 for t in curr_spike_times_msec]
        curr_spike_times_sec_ref = [t-time0_sec for t in curr_spike_times_sec]
        curr_spike_times_samples = [int(t*sfreq) for t in curr_spike_times_sec_ref] # convert to samples from sec
        spikes_zero_one_vec[curr_spike_times_samples] = 1
        spikes_matrix_all_groups = np.vstack((spikes_matrix_all_groups, spikes_zero_one_vec))
    raw = mne.io.RawArray(spikes_matrix_all_groups, info)
    return raw
