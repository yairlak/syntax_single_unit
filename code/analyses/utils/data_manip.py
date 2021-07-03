import os
import glob
import pickle
import sys
import numpy as np
import mne
from scipy import io
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
from .features import build_feature_matrix_from_metadata
from wordfreq import word_frequency, zipf_frequency
from utils.utils import probename2picks
from scipy.ndimage import gaussian_filter1d
import neo
import h5py


class DataHandler:
    def __init__(self, patient, data_type, filt,
                 probe_name=None, channel_name=None, channel_num=None,
                 feature_list=None):
        # MAKE SURE patient, data_type and filt are all lists
        if isinstance(patient, str):
            patient = [patient]
        if isinstance(data_type, str):
            data_type = [data_type]
        if isinstance(filt, str):
            filt = [filt]
        assert len(patient) == len(data_type) == len(filt)
        self.patient = patient
        self.data_type = data_type
        self.filter = filt
        self.probe_name = probe_name
        self.channel_name = channel_name
        self.channel_num = channel_num
        self.feature_list = feature_list

    def load_raw_data(self, scale_features=None, verbose=False):
        '''

        Parameters
        ----------
        scaling_method : TYPE, optional
            Which scaling method to use: 'standard' or 'robust'.
            If None then no scaling is performed. The default is None.
        verbose : TYPE, optional
            Verbosity. The default is False.

        Returns
        -------
        None.

        '''
        
        
        self.raws = []  # list of raw MNE objects
        for p, (patient, data_type, filt) in enumerate(zip(self.patient,
                                                           self.data_type,
                                                           self.filter)):
            # Load RAW object
            path2rawdata = f'../../Data/UCLA/{patient}/Raw/mne'
            fname_raw = '%s_%s_%s-raw.fif' % (patient, data_type, filt)

            raw_neural = mne.io.read_raw_fif(os.path.join(path2rawdata,
                                                          fname_raw),
                                             preload=True)
            # SAMPLING FREQUENCY
            self.sfreq = raw_neural.info['sfreq']
            
            # PICK
            picks = None
            if self.probe_name:
                picks = probename2picks(self.probe_name[p],
                                        raw_neural.ch_names,
                                        data_type)
            if self.channel_name:
                picks = self.channel_name[p]
            if self.channel_num:
                picks = self.channel_num[p]
            if verbose:
                print('picks:', picks)
            raw_neural.pick(picks)

            if self.feature_list:
                metadata_features = get_metadata_features(patient, data_type, self.sfreq)
                raw_features, self.feature_names,\
                    self.feature_info, self.feature_groups = \
                    get_raw_features(metadata_features, self.feature_list,
                                     len(raw_neural), self.sfreq)
                raw_neural.load_data()
                raw_neural = raw_neural.add_channels([raw_features],
                                                     force_update_info=True)
            if scale_features:
                if scale_features == 'standard':
                    scaler = StandardScaler()
                elif scale_features == 'robust':
                    scaler = RobustScaler()
                # raw_neural might already include feature channels:
                features_without_scaling=['is_first_word', 'is_first_phone']
                picks = mne.pick_types(raw_neural.info,
                                       misc=True,
                                       include=[], # include all but:
                                       exclude=features_without_scaling)
                print(f'{scale_features.capitalize()} scaling {len(picks)} FEATURE channels')
                
                scaled_data = scaler.fit_transform(raw_neural.copy().pick(picks).get_data().T)
                raw_neural._data[picks, :] = scaled_data.T
                
            self.raws.append(raw_neural)
            

        if verbose:
            print(self.raws)
            [print(raw.ch_names) for raw in self.raws]

    def epoch_data(self, level,
                   tmin=None, tmax=None, decimate=None, query=None,
                   block_type=None, scale_epochs=False, verbose=False,
                   smooth=None):
        '''
        Parameters
        ----------
        level : TYPE, optional
            DESCRIPTION. The default is None.
        tmin : TYPE, optional
            DESCRIPTION. The default is None.
        tmax : TYPE, optional
            DESCRIPTION. The default is None.
        decimate : TYPE, optional
            DESCRIPTION. The default is None.
        query : TYPE, optional
            DESCRIPTION. The default is None.
        block_type : TYPE, optional
            DESCRIPTION. The default is None.
        scale_epochs : TYPE, optional
            DESCRIPTION. The default is False.
        verbose : TYPE, optional
            DESCRIPTION. The default is False.
        smooth : float, optional
            Smoothing window size in miliseconds. The default is None.

        Returns
        -------
        None.

        '''

        self.epochs = []
        for p, (patient, data_type) in enumerate(zip(self.patient,
                                                     self.data_type)):
            print(f'Epoching {patient}, {data_type}, {level}')
            ##########
            events, event_id, metadata = get_events(patient, level, data_type, self.sfreq)
            ############
            # EPOCHING #
            ############
            # First epoch then filter if needed
            if verbose:
                print(self.raws[p].first_samp, events)
            if level == 'sentence_onset':
                tmin_, tmax_ = (-1.2, 3.5)
            elif level == 'sentence_offset':
                tmin_, tmax_ = (-3.5, 1.5)
            elif level == 'word':
                tmin_, tmax_ = (-1, 2)
            elif level == 'phone':
                tmin_, tmax_ = (-0.3, 1.2)
            epochs = mne.Epochs(self.raws[p], events, event_id, tmin_, tmax_,
                                metadata=metadata, baseline=None,
                                preload=True, reject=None)
            del events, event_id, metadata
            if any(epochs.drop_log):
                print('Dropped:', epochs.drop_log)

            if block_type == 'auditory':
                epochs = epochs['block in [2, 4, 6]']
            elif block_type == 'visual':
                epochs = epochs['block in [1, 3, 5]']
            # EXTEND METADATA
            epochs.metadata = extend_metadata(epochs.metadata)
            # QUERY
            if query:
                epochs = epochs[query]
            if verbose:
                print(query)
                print(epochs)
            # CROP
            if tmin and tmax:
                epochs = epochs.crop(tmin=tmin, tmax=tmax)
            # DECIMATE
            if decimate:
                epochs.decimate(decimate)
                self.sfreq = epochs.info['sfreq']

            # Separate neural data from features before pick and scale
            epochs_neural = epochs.copy().pick_types(seeg=True, eeg=True)
            if self.feature_list:
                epochs_features = epochs.copy().pick_types(misc=True)


            if smooth:
                width_sec = smooth/1000  # Gaussian-kernal width in [sec]
                print(f'smoothing data with {width_sec} sec window')
                data = epochs_neural.copy().get_data()
                for ch in range(data.shape[1]):  # over channels
                    for tr in range(data.shape[0]):  # over trials
                        time_series = data[tr, ch, :]
                        data[tr, ch, :] = gaussian_filter1d(
                            time_series, width_sec*self.sfreq)
                epochs_neural._data = data
            
            
            ############################
            # Robust Scaling Transform #
            ############################
            if scale_epochs:
                data = epochs_neural.get_data()
                n_trials, n_chs, n_times = data.shape
                for i_ch in range(n_chs):
                    vec = data[:, i_ch, :].reshape(-1, 1)
                    vec_scaled = StandardScaler().fit_transform(vec)
                    epochs_neural._data[:, i_ch, :] = \
                        vec_scaled.reshape(n_trials, n_times)

            

            if self.feature_list:
                # Put together neural and feature data
                epochs_neural.add_channels([epochs_features])
                # Hack to overcome MNE's possible bug in epochs.add_channels()
                epochs_neural.picks = np.concatenate((epochs_neural.picks,
                                                      epochs_features.picks))
            # APPEND
            self.epochs.append(epochs_neural)


def get_metadata_features(patient, data_type, sfreq):
    '''
    Generate metadata with features for patient

    Parameters
    ----------
    patient : int
        patient number.
    data_type : str
        micro/macro/spike.

    Returns
    -------
    metadata_features : TYPE
        DESCRIPTION.

    '''

    _, _, metadata_phone = get_events(patient, 'phone', data_type, sfreq)
    _, _, metadata_word = get_events(patient, 'word', data_type, sfreq)
    metadata_audio = extend_metadata(metadata_phone)
    metadata_visual = metadata_word.query('block in [1, 3, 5]')
    metadata_visual = extend_metadata(metadata_visual)
    metadata_features = pd.concat([metadata_audio, metadata_visual], axis=0)
    metadata_features = metadata_features.sort_values(by='event_time')
    return metadata_features


def get_raw_features(metadata_features, feature_list, num_time_samples, sfreq):
    # CREATE DESIGN MATRIX
    X_features, feature_names, feature_info, feature_groups = \
        build_feature_matrix_from_metadata(metadata_features, feature_list)
    _, num_features = X_features.shape
    times_sec = metadata_features['event_time'].values
    times_samples = (times_sec * sfreq).astype(int)
    # add 10sec for RF
    X = np.zeros((num_time_samples, num_features))
    X[times_samples, :] = X_features
    # STANDARIZE THE FEATURE MATRIX #
    scaler = StandardScaler()
    # EXCEPT FOR WORD AND SENTENCE ONSET
    IX_sentence_onset = feature_names.index('is_first_word')
    sentence_onset = X[:, IX_sentence_onset]
    if 'is_first_phone' in feature_names:
        IX_word_onset = feature_names.index('is_first_phone')
        word_onset = X[:, IX_word_onset]
    X = scaler.fit_transform(X)
    X[:, IX_sentence_onset] = sentence_onset
    if 'is_first_phone' in feature_names:
        X[:, IX_word_onset] = word_onset
    # MNE-ize feature data
    ch_types = ['misc'] * len(feature_names)
    info = mne.create_info(ch_names=feature_names,
                           ch_types=ch_types,
                           sfreq=sfreq)
    raw_features = mne.io.RawArray(X.T, info)
    return raw_features, feature_names, feature_info, feature_groups


def get_events(patient, level, data_type, sfreq, verbose=False):
    blocks = range(1, 7)
    #sfreq = 1000  # Data types downsamplled to 1000Hz by generate_mne_raw.py

    #TODO: add log to power
    
    if verbose:
        print('Reading logs from experiment...')
    path2log = os.path.join('..', '..', 
                            'Data', 'UCLA', patient, 'Logs')
    log_all_blocks = {}
    for block in blocks:
        log = read_log(block, path2log)
        log_all_blocks[block] = log
    if verbose:
        print('Preparing meta-data')
    metadata = prepare_metadata(log_all_blocks, data_type, level)

    # First column of events object
    times_in_sec = sorted(metadata['event_time'].values)
    min_diff_sec = np.min(np.diff(times_in_sec))
    if verbose:
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

    # HACK: since spike sorting was done with CSC*.ncs files that were not merged
    # Hence, timing should be shifted by the length of the first ncs files (suffix0)
    # if patient == 'patient_479_25' and data_type=='spike':
    #     events[:, 0] -= int(117.647 * sfreq)
    return events, event_id, metadata


def generate_mne_raw(data_type, from_mat, path2rawdata, sfreq_down):
    
    assert not (data_type == 'spike' and from_mat)
    
    # Path to data
    path2data = os.path.join(path2rawdata, data_type)
    if from_mat:
        path2data = os.path.join(path2data, 'mat')
    print(f'Loading data from: {path2data}')
    
    # Extract raw data
    if data_type == 'spike':
        channel_data, ch_names, sfreq = get_data_from_combinato(path2data)
    else:
        if from_mat:
            channel_data, ch_names, sfreq = get_data_from_mat(data_type, path2data)
        else:
            raw = get_data_from_ncs_or_ns(data_type, path2data, sfreq_down)
            return raw
    #print(f'Shape channel_data: {channel_data.shape}')
    n_channels = channel_data.shape[0]
    ch_types = ['seeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(channel_data, info)

    return raw


def get_data_from_combinato(path2data):

    sfreq = 1000

    print('Loading spike cluster data')
    
    CSC_folders = glob.glob(os.path.join(path2data, 'CSC?/')) + \
                  glob.glob(os.path.join(path2data, 'CSC??/')) + \
                  glob.glob(os.path.join(path2data, 'CSC???/'))
    
    reader = neo.io.NeuralynxIO(path2data)        
    channel_tuples = reader.header['signal_channels']
    time0, timeend = reader.global_t_start, reader.global_t_stop    
    print(f'time0 = {time0}, timeend = {timeend}')
    
    ch_names, spike_times_samples = [], []
    for CSC_folder in CSC_folders:
        channel_num = int(CSC_folder.split('CSC')[-1].strip('/'))
        probe_name = [t[0] for t in channel_tuples if t[1]==channel_num-1]
        # print(channel_num, probe_name)
        if probe_name:
            probe_name = probe_name[0]
        else:
            probe_name = '?'
            print(f'Unable to identify probe name for channel {channel_num}')
        spikes, group_names = load_combinato_sorted_h5(path2data, channel_num,
                                                       probe_name)
        #print(group_names, len(group_names), len(spikes))
        assert len(group_names)==len(spikes)

        if len(spikes) > 0:
            ch_names.extend(group_names)
            for groups, curr_spike_times_msec in enumerate(spikes):
                curr_spike_times_samples = [int(t*sfreq/1e3) for t in curr_spike_times_msec] # convert to samples from sec
                spike_times_samples.append(curr_spike_times_samples)
        else:
            print(f'No spikes in channel: {channel_num}')

    # ADD to array
    #print(ch_names)
    #print('ch_names', 'spikes', len(ch_names), len(spike_times_samples))
    num_groups = len(spike_times_samples)
    channel_data = np.zeros((num_groups, int(1e3*(timeend- time0 + 1))))
    for i_st, st in enumerate(spike_times_samples):
        st = (np.asarray(st) - time0*1e3).astype(int)
        channel_data[i_st, st] = 1
    
    return channel_data, ch_names, sfreq


def get_data_from_mat(data_type, path2data):
     
    if data_type == 'microphone':
        CSC_files = glob.glob(os.path.join(path2data, 'MICROPHONE.mat'))
        assert len(CSC_files) == 1
    else:
        CSC_files = glob.glob(os.path.join(path2data, 'CSC?.mat')) + \
                    glob.glob(os.path.join(path2data, 'CSC??.mat')) + \
                    glob.glob(os.path.join(path2data, 'CSC???.mat'))
        assert len(CSC_files) > 0
        CSC_nums = [int(os.path.basename(s)[3:-4]) for s in CSC_files]
        IX = np.argsort(CSC_nums)
        CSC_files = np.asarray(CSC_files)[IX]

    channel_data, ch_names = [], []
    for i_ch, CSC_file in enumerate(CSC_files):
        curr_channel_data = io.loadmat(CSC_file)    
        sfreq = int(1e3/curr_channel_data['samplingInterval'])
        channel_data.append(curr_channel_data['data'])
        if 'channelName' in curr_channel_data:
            ch_name = curr_channel_data['channelName']
        else:
            ch_name = os.path.basename(CSC_file)[:-4]
        ch_names.append(ch_name)
        print(f'Processing file: {ch_name} ({i_ch+1}/{len(CSC_files)}), sfreq = {sfreq} Hz')
    channel_data = np.vstack(channel_data)

    return channel_data, ch_names, sfreq


def get_data_from_ncs_or_ns(data_type, path2data, sfreq_down):
    if data_type == 'microphone':
        # Assumes that if data_type is microphone 
        # Then the recording system is Neurlanyx.
        # Otherwise, The flag --from-mat should be used
        recording_system = 'Neuralynx'
    else:
        recording_system = identify_recording_system(path2data)
    
    if recording_system == 'Neuralynx':
        reader = neo.io.NeuralynxIO(dirname=path2data)
        time0, timeend = reader.global_t_start, reader.global_t_stop
        sfreq = reader._sigs_sampling_rate
        print(f'Neural files: Start time {time0}, End time {timeend}')
        print(f'Sampling rate [Hz]: {sfreq}')
        blks = reader.read(lazy=True)
        channels = reader.header['signal_channels']
        n_channels = len(channels)
        ch_names = [channel[0] for channel in channels]
        channel_nums = [channel[1] for channel in channels]
        print('Number of channel %i: %s'
                      % (len(ch_names), ch_names))

        raws = []
        for i_segment, segment in enumerate(blks[0].segments):
            print(f'Segment - {i_segment+1}/{len(blks[0].segments)}')
            asignal = segment.analogsignals[0].load()
            
            raw_channels = []
            for i_ch in range(n_channels):
                info = mne.create_info(ch_names=[ch_names[i_ch]],
                                       sfreq=sfreq, ch_types='seeg')
                raw = mne.io.RawArray(np.asarray(asignal[:, i_ch]).T, info, verbose=False)
                if data_type != 'microphone':
                    # Downsample
                    if raw.info['sfreq'] > sfreq_down:
                        print('Resampling data %1.2f -> %1.2f' % (raw.info['sfreq'], sfreq_down))
                        raw = raw.resample(sfreq_down, npad='auto')
                if i_ch == 0:
                    raw_channels = raw.copy()
                else:
                    raw_channels.add_channels([raw])
            raws.append(raw_channels)
        del blks
        raws = mne.concatenate_raws(raws)
    elif recording_system == 'BlackRock':
        reader = io.BlackrockIO(path2data)
        sfreq = reader.header['unit_channels'][0][-1] # FROM FILE
        print(sfreq)
        print(dir(reader))
        raise('Implementation error')

    return raws


def identify_recording_system(path2data):
    neural_files = glob.glob(os.path.join(path2data, '*.n*'))
    if len(neural_files)>1:
        recording_system = 'Neuralynx'
        assert neural_files[0][-3:] == 'ncs'
    elif len(neural_files)==1:
        recording_system = 'BlackRock'
        assert len(neural_files[0][-3:]) == 2
    else:
        print(f'No neural files found: {path2data}')
        raise()

    return recording_system


def load_combinato_sorted_h5(path2data, channel_num, probe_name):
    target_types = [2] # -1: artifact, 0: unassigned, 1: MU, 2: SU

    spike_times_msec = []; group_names = []
    h5_files = glob.glob(os.path.join(path2data, 'CSC' + str(channel_num), 'data_*.h5'))
    if len(h5_files) == 1:
        filename = h5_files[0]
        f_all_spikes = h5py.File(filename, 'r')

        for sign in ['neg', 'pos']:
        #for sign in ['neg']:
            filename_sorted = glob.glob(os.path.join(path2data, 'CSC' + str(channel_num), 'sort_' + sign + '_yl2', 'sort_cat.h5'))
            if len(filename_sorted) == 1:
                f_sort_cat = h5py.File(filename_sorted[0], 'r')
                group_numbers = []
                try:
                    classes =  f_sort_cat['classes'][:]
                    index = f_sort_cat['index'][:]
                    matches = f_sort_cat['matches'][:]
                    groups = f_sort_cat['groups'][:]
                    group_numbers = set([g[1] for g in groups])
                    types = f_sort_cat['types'][:] # -1: artifact, 0: unassigned, 1: MU, 2: SU
                except:
                    print('Something is wrong with %s, %s' % (sign, filename_sorted[0]))

                # For each group, generate a list with all spike times and append to spike_times
                for g in list(group_numbers):
                    IXs = []
                    type_of_curr_group = [t_ for (g_, t_) in types if g_ == g]
                    if len(type_of_curr_group) == 1:
                        type_of_curr_group = type_of_curr_group[0]
                    elif not any([t in target_types for t in type_of_curr_group]):
                        print(f'No target type was found for group {g}')
                        continue
                    else:
                        raise ('issue with types: more than one group assigned to a type')
                    # if type_of_curr_group>0: # ignore artifact and unassigned groups
                    if type_of_curr_group in target_types: # Single-unit (SU) only
                        print(f'found cluster in {probe_name}, channel {channel_num}, group {g}')
                        # Loop over all spikes
                        for i, c in enumerate(classes):
                            # check if current cluster in group
                            g_of_curr_cluster = [g_ for (c_, g_) in groups if c_ == c]
                            if len(g_of_curr_cluster) == 1:
                                g_of_curr_cluster = g_of_curr_cluster[0]
                            else:
                                raise('issue with groups: more than one group assigned to a cluster')
                            # if curr spike is in a cluster of the current group
                            if g_of_curr_cluster == g:
                                curr_IX = index[i]
                                IXs.append(curr_IX)

                        curr_spike_times = f_all_spikes[sign]['times'][:][IXs]
                        spike_times_msec.append(curr_spike_times)
                        #print(sign[0], g, channel_num, probe_name)
                        group_names.append(sign[0] + '_g' + str(g) + '_' + str(channel_num)+ '_' + probe_name)
            else:
                print('%s was not found!' % os.path.join(path2data, 'micro', 'CSC_ncs', 'CSC' + str(channel_num), 'sort_' + sign + '_yl2', 'sort_cat.h5'))

    else:
        print('None or more than a single combinato h5 was found')

    return spike_times_msec, group_names





def add_event_to_metadata(metadata, event_time, sentence_number, sentence_string, word_position, word_string, pos, num_words, last_word):
    metadata['event_time'].append(event_time)
    metadata['sentence_number'].append(sentence_number)
    metadata['sentence_string'].append(sentence_string)
    metadata['word_position'].append(word_position)
    metadata['word_string'].append(word_string)
    metadata['pos'].append(pos)
    metadata['num_words'].append(num_words)
    metadata['last_word'].append(last_word)
    return metadata


def convert_to_mne_python(data, events, event_id, electrode_info, metadata, sfreq_data, tmin, tmax):
    '''

    :param data:
    :param events:
    :param event_id:
    :param electrode_labels:
    :param metadata:
    :param sfreq_data:
    :param tmin:
    :param tmax:
    :return:
    '''
    channel_names = [s[0][0]+'-'+s[3][0] for s in electrode_info['labels']]
    regions = set([s[3][0] for s in electrode_info['labels']])
    print('Regions: ' + ' '.join(regions))
    montage = mne.channels.Montage(electrode_info['coordinates'], channel_names, 'misc', range(len(channel_names)))
    # fig_montage = montage.plot(kind='3d')
    num_channels = data.shape[0]
    ch_types = ['seeg' for s in range(num_channels)]
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq_data, ch_types=ch_types)#, montage=montage)
    raw = mne.io.RawArray(data, info)

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, metadata=metadata, baseline=None,
                        preload=False)
    return info, epochs



def create_events_array(metadata):
    '''

    :param metadata: (pandas dataframe) num_words X num_features; all words across all stimuli
    :param params: (object) general parameters
    :return:
    '''

    # First column of events object
    curr_times = np.expand_dims(metadata['event_time'].values, axis=1)

    # Second column
    second_column = np.zeros((len(curr_times), 1))

    # Third column
    event_numbers = range(len(curr_times))  # For each block, the event_ids are ordered within a range of 100 numbers block1: 101-201, block2: 201-300, etc.
    event_numbers = np.expand_dims(event_numbers, axis=1)

    # EVENT object: concatenate all three columns together (then change to int and sort)
    events = np.hstack((curr_times, second_column, event_numbers))
    events = events.astype(int)
    sort_IX = np._argsort(events[:, 0], axis=0)
    events = events[sort_IX, :]

    # EVENT_ID dictionary: mapping block names to event numbers
    event_id = dict([(str(event_type_name), event_number[0]) for event_type_name, event_number in zip(event_numbers, event_numbers)])


    return events, event_id




def load_neural_data(patient, data_type, filt, level,
                     probe_name=None, channel_name=None, channel_num=None,
                     tmin=None, tmax=None, decimate=None,
                     query=None, block_type=None,
                     scale_epochs=False, verbose=False):
    '''
    Parameters
    ----------
    patient : TYPE
        DESCRIPTION.
    data_type : TYPE
        DESCRIPTION.
    filt : TYPE
        DESCRIPTION.
    level : TYPE
        DESCRIPTION.
    probe_name : TYPE, optional
        DESCRIPTION. The default is None.
    channel_name : TYPE, optional
        DESCRIPTION. The default is None.
    channel_num : TYPE, optional
        DESCRIPTION. The default is None.
    tmin : TYPE, optional
        DESCRIPTION. The default is None.
    tmax : TYPE, optional
        DESCRIPTION. The default is None.
    decimate : TYPE, optional
        DESCRIPTION. The default is None.
    query : TYPE, optional
        DESCRIPTION. The default is None.
    block_type : TYPE, optional
        DESCRIPTION. The default is None.
    scale_epochs : TYPE, optional
        DESCRIPTION. The default is False.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    epochs_list : list
        List of epochs

    '''
    import mne
    from utils.utils import probename2picks  # pick_responsive_channels
    # from utils.read_logs_and_features import extend_metadata

    if isinstance(patient, str):  # in case a patient list is not provided
        patients = [patient]
    else:
        patients = patient
    if isinstance(data_type, str):
        data_types = [data_type]
    else:
        data_types = data_type
    if isinstance(filt, str):
        filters = [filt]
    else:
        filters = filt

    epochs_list = []
    for p, (patient, data_type, filt) in enumerate(zip(patients,
                                                       data_types,
                                                       filters)):
        settings = Settings(patient)
        ###################
        # Load RAW object #
        ###################
        fname_raw = '%s_%s_%s-raw.fif' % (patient, data_type, filt)
        raw = mne.io.read_raw_fif(os.path.join(settings.path2rawdata,
                                               fname_raw), preload=False)
        if verbose:
            print(raw, raw.ch_names)
        ##########
        # EVENTS #
        ##########
        events, event_id, metadata = get_events(patient, level, data_type)
        ############
        # EPOCHING #
        ############
        # First epoch then filter if needed
        if verbose:
            print(raw.first_samp, events)
        if level == 'sentence_onset':
            tmin, tmax = (-1.2, 3.5)
        elif level == 'sentence_offset':
            tmin, tmax = (-3.5, 1.5)
        elif level == 'word':
            tmin, tmax = (-1, 2)
        elif level == 'phone':
            tmin, tmax = (-0.3, 1.2)
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                            metadata=metadata, baseline=None,
                            preload=True, reject=None)
        if any(epochs.drop_log):
            print('Dropped:')
            print(epochs.drop_log)
        ############################
        # Robust Scaling Transform #
        ############################
        if scale_epochs:
            data = epochs.copy().get_data()
            for ch in range(data.shape[1]):
                transformer = RobustScaler().fit(np.transpose(data[:, ch, :]))
                epochs._data[:, ch, :] = \
                    np.transpose(
                        transformer.transform(np.transpose(data[:, ch, :])))

        if block_type == 'auditory':
            epochs = epochs['block in [2, 4, 6]']
        elif block_type == 'visual':
            epochs = epochs['block in [1, 3, 5]']
        # EXTEND METADATA
        epochs.metadata = extend_metadata(epochs.metadata)
        # QUERY
        if query:
            epochs = epochs[query]
        if verbose:
            print(query)
            print(epochs)
        # CROP
        if tmin and tmax:
            epochs = epochs.crop(tmin=tmin, tmax=tmax)
        # PICK
        picks = None
        if probe_name:
            probe_name = probe_name[p]
            picks = probename2picks(probe_name, epochs.ch_names, data_type)
        if channel_name:
            channel_names = channel_name[p]
            picks = channel_names
        if channel_num:
            picks = channel_num[p]
        if verbose:
            print('picks:', picks)
        epochs.pick(picks)
        # DECIMATE
        if decimate:
            epochs.decimate(decimate)
        # APPEND
        epochs_list.append(epochs)

    return epochs_list


def read_log(block, path2log, log_name_beginning='new_with_phones_events_log_in_cheetah_clock_part'):
    '''

    :param block: (int) block number
    :return: events (dict) with keys for event_times, block, phone/word/stimulus info
    '''
    log_fn = log_name_beginning + str(block) + '.log'
    with open(os.path.join(path2log, log_fn)) as f:
        lines = [l.strip('\n').split(' ') for l in f]
    
    events = {}
    if block in [2, 4, 6]:
        lines = [l for l in lines if l[1]=='PHONE_ONSET']
        events['event_time'] = [l[0] for l in lines]
        events['block'] = len(events['event_time']) * [block]
        events['is_first_phone'] = [int(l[2]) for l in lines]
        events['phone_position'] = [int(l[3]) for l in lines]
        events['phone_string'] = [l[6] for l in lines]
        events['word_position'] = [int(l[4]) for l in lines]
        events['word_string'] = [l[7] for l in lines]
        events['stimulus_number'] = [int(l[5]) for l in lines]

    elif block in [1, 3, 5]:
        lines = [l for l in lines if l[1] == 'DISPLAY_TEXT' and l[2] != 'OFF']
        events['event_time'] = [l[0] for l in lines]
        events['block'] = len(events['event_time']) * [block]
        events['is_first_phone'] = len(events['event_time']) * [0] # not relevant for visual blocks
        events['phone_position'] = len(events['event_time']) * [0] # not relevant for visual blocks
        events['phone_string'] = len(events['event_time']) * ['']  # not relevant for visual blocks
        events['word_position'] = [int(l[4]) for l in lines]
        events['word_string'] = [l[5] for l in lines]
        events['stimulus_number'] = [int(l[3]) for l in lines]

    return events


def prepare_metadata(log_all_blocks, data_type, level):
    '''
    :param log_all_blocks: list len = #blocks
    :param features: numpy
    :return: metadata: list
    '''
    word_ON_duration = 200 # [msec]
    word2features, word2features_new = load_word_features()
    #print(word2features_new)
    num_blocks = len(log_all_blocks)

    # Create a dict with the following keys:
    keys = ['chronological_order', 'event_time', 'block', 'is_first_phone', 'phone_position', 'phone_string', 'stimulus_number',
            'word_position', 'word_string', 'pos', 'dec_quest', 'grammatical_number', 'wh_subj_obj',
            'word_length', 'sentence_string', 'sentence_length', 'last_word', 'morpheme', 'morpheme_type', 'word_type', 'word_freq', 'word_zipf']
    metadata = dict([(k, []) for k in keys])

    cnt = 1
    events_all_blocks = []
    for block, curr_block_events in log_all_blocks.items():
        for i in range(len(curr_block_events['event_time'])):
            sn = int(curr_block_events['stimulus_number'][i])
            wp = int(curr_block_events['word_position'][i])
            if wp == -1: wp = 0
            #print(sn, wp)
            #print(word2features_new[sn])
            metadata['stimulus_number'].append(sn)
            metadata['word_position'].append(wp)
            metadata['chronological_order'].append(cnt); cnt += 1
            #if data_type == 'macro' and settings.recording_device == 'BlackRock': # If micro/macro recorded with different devices
            #    time0 = settings.time0_macro
            #else:
            #    time0 = settings.time0
            #metadata['event_time'].append((int(curr_block_events['event_time'][i]) - time0) / 1e6)
            metadata['event_time'].append(int(float(curr_block_events['event_time'][i])) / 1e6)
            metadata['block'].append(curr_block_events['block'][i])
            is_first_phone = curr_block_events['is_first_phone'][i]
            if is_first_phone==-1: is_first_phone=0
            metadata['is_first_phone'].append(is_first_phone)
            phone_pos = int(curr_block_events['phone_position'][i])
            metadata['phone_position'].append(phone_pos)
            metadata['phone_string'].append(curr_block_events['phone_string'][i])
            word_string = curr_block_events['word_string'][i]
            if word_string[-1] == '?' or word_string[-1] == '.':
                word_string = word_string[0:-1]
            if word_string == '-': word_string = ''
            if block in [2, 4, 6] and word_string:
                word_string = word_string.lower()
                if wp == 1:
                    word_string = word_string.capitalize()
            metadata['word_string'].append(word_string)
            word_string = word_string.lower()
            word_freq = word_frequency(word_string, 'en')
            word_zipf = zipf_frequency(word_string, 'en')
            #print(word_string, type(word_freq), type(word_zipf))
            # ADD FEATURES FROM XLS FILE
            sentence_onset = wp==1 and ((curr_block_events['phone_string'][i] != 'END_OF_WAV' and phone_pos==1) or (phone_pos==0))
            middle_word_onset = wp!=1 and ((curr_block_events['phone_string'][i] != 'END_OF_WAV' and phone_pos>1 and is_first_phone) or (curr_block_events['block'][i] in [1,3,5]))
            middle_phone = (curr_block_events['phone_string'][i] != 'END_OF_WAV' and (not is_first_phone) and (curr_block_events['block'][i] in [2,4,6]))
            if sentence_onset: # ADD WORD AND- SENTENCE-LEVEL FEATURES
                metadata['sentence_string'].append(word2features_new[sn][wp]['sentence_string'])
                metadata['sentence_length'].append(word2features_new[sn][wp]['sentence_length'])
                metadata['word_length'].append(word2features_new[sn][wp]['word_length'])
                metadata['dec_quest'].append(word2features_new[sn][wp]['dec_quest'])
                metadata['grammatical_number'].append(word2features_new[sn][wp]['grammatical_number'])
                metadata['pos'].append(word2features_new[sn][wp]['pos'])
                metadata['wh_subj_obj'].append(word2features_new[sn][wp]['wh_subj_obj'])
                metadata['morpheme'].append(word2features[word_string][0])
                metadata['morpheme_type'].append(int(word2features[word_string][1]))
                metadata['word_type'].append(word2features[word_string][2])
                metadata['last_word'].append(metadata['sentence_length'][-1] == metadata['word_position'][-1])
                metadata['word_freq'].append(word_freq)
                metadata['word_zipf'].append(word_zipf)
            elif middle_word_onset: # ADD WORD-LEVEL FEATURES
                metadata['sentence_string'].append(word2features_new[sn][wp]['sentence_string'])
                metadata['sentence_length'].append(word2features_new[sn][wp]['sentence_length'])
                metadata['word_length'].append(word2features_new[sn][wp]['word_length'])
                metadata['dec_quest'].append(0)
                metadata['grammatical_number'].append(word2features_new[sn][wp]['grammatical_number'])
                metadata['pos'].append(word2features_new[sn][wp]['pos'])
                metadata['wh_subj_obj'].append(0)
                metadata['morpheme'].append(word2features[word_string][0])
                metadata['morpheme_type'].append(int(word2features[word_string][1]))
                metadata['word_type'].append(word2features[word_string][2])
                metadata['last_word'].append(metadata['sentence_length'][-1] == metadata['word_position'][-1])
                metadata['word_freq'].append(word_freq)
                metadata['word_zipf'].append(word_zipf)
            elif middle_phone:  # NO WORD/SENTENCE-LEVEL FEATURES
                metadata['sentence_string'].append(
                    word2features_new[sn][wp]['sentence_string'])
                metadata['sentence_length'].append(
                    word2features_new[sn][wp]['sentence_length'])
                metadata['word_length'].append(0)
                metadata['dec_quest'].append(0)
                metadata['grammatical_number'].append(0)
                metadata['pos'].append('')
                metadata['wh_subj_obj'].append(0)
                metadata['morpheme'].append('')
                metadata['morpheme_type'].append('')
                metadata['word_type'].append('')
                metadata['last_word'].append(False)
                metadata['word_freq'].append(0)
                metadata['word_zipf'].append(0)
            elif curr_block_events['phone_string'][i] == 'END_OF_WAV':
                metadata['sentence_string'].append(
                    word2features_new[sn][wp]['sentence_string'])
                metadata['sentence_length'].append(
                    word2features_new[sn][wp]['sentence_length'])
                metadata['word_length'].append(0)
                metadata['dec_quest'].append(0)
                metadata['grammatical_number'].append(0)
                metadata['pos'].append('')
                metadata['wh_subj_obj'].append(0)
                metadata['morpheme'].append('')
                metadata['morpheme_type'].append('')
                metadata['word_type'].append('')
                metadata['last_word'].append(False)
                metadata['word_freq'].append(0)
                metadata['word_zipf'].append(0)
                metadata['phone_position'][-1] = 0
            else:
                raise('Unknown log value')
            # SINCE ONLY IN THE AUDIO LOGS THERE'S END-OF-WAV (WORD_POSITION=0)
            # ANOTHER ROW FOR END-OF-SENTENCE IS ADDED FOR VISUAL BLOCKS
            # (OFFSET OF LAST WORD)
            if metadata['last_word'][-1] and metadata['block'][-1] in [1, 3, 5]:
                metadata['chronological_order'].append(cnt)
                cnt += 1
                t = metadata['event_time'][-1] + word_ON_duration*1e-3
                metadata['event_time'].append(t)
                metadata['block'].append(curr_block_events['block'][i])
                metadata['is_first_phone'].append(0)
                metadata['phone_position'].append(0)
                metadata['phone_string'].append('')
                metadata['stimulus_number'].append(
                    int(curr_block_events['stimulus_number'][i]))
                metadata['word_position'].append(0)
                metadata['word_string'].append('.')
                metadata['pos'].append('')
                metadata['morpheme'].append('')
                metadata['morpheme_type'].append('')
                metadata['word_type'].append('')
                metadata['word_freq'].append(0)
                metadata['word_zipf'].append(0)
                metadata['sentence_string'].append(
                    word2features_new[sn][wp]['sentence_string'])
                metadata['sentence_length'].append(
                    word2features_new[sn][wp]['sentence_length'])
                metadata['word_length'].append(0)
                metadata['dec_quest'].append(0)
                metadata['grammatical_number'].append(0)
                metadata['wh_subj_obj'].append(0)
                metadata['last_word'].append(False)

    metadata = pd.DataFrame(metadata)
    if level == 'sentence_onset':
        metadata = metadata.loc[((metadata['block'].isin([1, 3, 5])) &
                                 (metadata['word_position'] == 1)) |
                                ((metadata['block'].isin([2, 4, 6])) &
                                 (metadata['word_position'] == 1) &
                                 (metadata['phone_position'] == 1))]
        # tmin, tmax = (-1, 3.5)
    elif level == 'sentence_offset':
        # metadata = metadata.loc[(metadata['word_position'] == 0)]
        metadata = metadata.loc[(metadata['last_word'] == 1)]
        # tmin, tmax = (-3.5, 1.5)
    elif level == 'word':
        # filter metadata to word-onset events, first-phone==-1 (visual blocks)
        metadata = metadata.loc[((metadata['is_first_phone'] == 1) &
                                 (metadata['block'].isin([2, 4, 6]))) |
                                ((metadata['block'].isin([1, 3, 5])) &
                                 (metadata['word_position'] > 0))] 
        # tmin, tmax = (-0.6, 1.5)
    elif level == 'phone':
        # filter metadata to only phone-onset events in auditory blocks
        metadata = metadata.loc[(metadata['block'].isin([2, 4, 6])) &
                                (metadata['phone_position'] > 0)]
        # tmin, tmax = (-0.3, 1.2)
    elif level == 'block':
        metadata = metadata.loc[(metadata['chronological_order'] == 1) |
                                (metadata['chronological_order'] == 509) |
                                (metadata['chronological_order'] == 2127) |
                                (metadata['chronological_order'] == 2635) |
                                (metadata['chronological_order'] == 4253) |
                                (metadata['chronological_order'] == 4761)]
    else:
        raise('Unknown level type (sentence_onset/sentence_offset/word/phone)')
    metadata.sort_values(by='event_time')
    return metadata


def extend_metadata(metadata):
    ''' Add columns to metadata
    '''
    metadata = metadata.rename(columns={'last_word': 'is_last_word'})
     # FIX ORTHOGRAPHIC MISTAKES
    metadata['word_string'] = metadata['word_string'].replace({'excercised':'exercised', 'heared':'heard', 'streched':'stretched'})
    
    # TENSE
    # LAST LETTER OF POS OF VERBS INDICATE THE TENSE (D - past, P - present, F - future, V - passive, I - infinitive-like, G - ing)
    poss = metadata['pos']
    tenses = []
    dict_tense = {'D':'past', 'P':'present', 'F':'future', 'V':'passive', 'I':'inf_like', 'G':'ing'}
    for i_pos, pos in enumerate(poss):
        # print(i_pos, pos)
        if pos.startswith('VB'):
            tense = dict_tense[pos[-1]]
            if tense == 'passive': tense = 'past' # HACK: all passive forms are in past
        else: # not a verb
            tense = ''
        tenses.append(tense)
    metadata['tense'] = tenses
    
    # POS SIMPLE
    pos = metadata['pos']
    # lump together all verbs (VBTRD, VBTRP, VBUEP,..)
    pos = ['VB' if p.startswith('VB') else p for p in pos]
    # lump together all nouns (NN, NNS)
    pos = ['NN' if p.startswith('NN') else p for p in pos]
    # lump together less frequent POS
    pos = ['OTHER' if p in ['JJ', 'RB'] else p for p in pos]
    metadata['pos_simple'] = pos

    # MORPHOLOGICAL COMPLEXITY
    morph_complex = [1 if m in ['d', 'ed', 'y', 'es', 'ing','s'] else 0 for m in metadata['morpheme']]
    metadata['morph_complex'] = morph_complex
  
    # IS FIRST WORD (LAST_WORD ALREADY IN METADATA)
    is_first_word = []
    for b, wp, ifp in zip(metadata['block'],
                          metadata['word_position'],
                          metadata['is_first_phone']):
        ifw = 0
        if (b in [1, 3, 5]) and wp == 1:
            ifw = 1
        elif (b in [2, 4, 6]) and wp == 1 and ifp == 1:
            ifw = 1
        is_first_word.append(ifw)
    metadata['is_first_word'] = is_first_word


    # EMBEDDING
    stim_numbers_with_that = [] # LIST OF TUPLES (STIM_NUM, WORD_POSITION_OF_THAT)
    for IX_word, w in enumerate(metadata['word_string']): # FIND STIMULUS NUMBER WITH THAT:
        if w == 'that':
            stim_numbers_with_that.append((metadata['stimulus_number'].tolist()[IX_word], metadata['word_position'].tolist()[IX_word]))
    embedding = [] # GENERATE A LIST OF VALUES: 1 - IN EMBEDDING, 0 - IN MAIN
    for curr_sn, curr_wp in zip(metadata['stimulus_number'], metadata['word_position']):
        is_in_embedding = any([1 if (curr_sn == sn and (curr_wp>=wp or curr_wp==-1)) else 0 for (sn, wp) in stim_numbers_with_that])
        #print(curr_sn, curr_wp, is_in_embedding)
        embedding.append(is_in_embedding)
    metadata['embedding'] = embedding

    # SEMANTIC FEATURES
    fn_glove = '../../Paradigm/small_glove.twitter.27B.25d.txt'

    glove = load_glove_model(fn_glove)
    #print(sorted(glove.keys()))
    X = []
    for i_w, w in enumerate(metadata['word_string']):
        if list(metadata['word_length'])[i_w]>1:
            vec = glove[w.lower()]
        else:
            vec = np.zeros(25)
        X.append(vec)
    metadata['semantic_features'] = X            
    
    # PHONOLOGICAL FEATURES
    phones = metadata['phone_string']
    fn_phonologica_features = 'features/phone.csv'
    df_phonological_features = pd.read_csv(fn_phonologica_features)
    phonological_features = list(df_phonological_features)
    phonological_features.remove('PHONE')
    # for phonological_feature in phonological_features:
    #     print(phonological_feature)
    feature_values = []
    for ph in phones:
        if ph and ph not in [-1, 'END_OF_WAV']:
            ph = ''.join([s for s in ph if not s.isdigit()]) # remove digits at the end if exist
            # feature_value = df_phonological_features.loc[df_phonological_features['PHONE'] == ph][phonological_feature]
            feature_value = df_phonological_features.loc[df_phonological_features['PHONE'] == ph]
            feature_values.append(feature_value.values[0][1::])
        else:
            feature_values.append(np.zeros((1, len(phonological_features))))
    metadata['phonological_features'] = feature_values
    # feature_values = np.vstack(feature_values)
    # feature_values = pd.DataFrame(data=feature_values, columns=phonological_features)
    # metadata = pd.concat((metadata, feature_values), axis=1)
    
    return metadata


def load_glove_model(glove_file):
    """
    :param glove_file: embeddings_path: path of glove file.
    :return: glove model
    """

    with open(glove_file, 'r', encoding='utf-8') as f:
        vectors = f.readlines()
    model = {}
    for line in vectors:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        model[word] = embedding
    return model


def get_sentences_start_end_length(SENTENCE_NUM_ORDER, settings):
    # Load text containing all sentences
    with open(os.path.join(settings.path2stimuli, settings.stimuli_text_file), 'r') as f:
        stimuli_sentences = f.readlines()
    sentences_length = [len(s.split(' ')) for s in stimuli_sentences]
    IX = [i-1 for i in SENTENCE_NUM_ORDER] # shift to count from zero
    sentences_length = np.asarray(sentences_length)[IX] #reorder array according to the (random) order of current block
    sentences_end = np.cumsum(sentences_length)
    sentences_start = [e-l+1 for (e, l) in zip(sentences_end, sentences_length)]

    sentences_length = dict(zip(range(1, len(sentences_length) + 1, 1), sentences_length))
    sentences_end = dict(zip(range(1, len(sentences_end) + 1, 1), sentences_end))
    sentences_start = dict(zip(range(1, len(sentences_start) + 1, 1), sentences_start))

    return sentences_start, sentences_end, sentences_length


def load_features(settings):
    import pandas

    # Read features file ('xlsx')
    sheet = pandas.read_excel(os.path.join(settings.path2stimuli, settings.features_file))
    headers = sheet.columns
    fields = []
    for i, header in enumerate(headers):
        fields.append(sheet[header].values)
    features = {'headers': headers, 'fields': fields}

    return features


def extract_comparison(comparison_list, features, settings, preferences):
    trial_numbers = features['fields'][0][1::]
    stimuli = features['fields'][1][1::]
    features = features['fields'][2::]
    contrast_names = comparison_list['fields'][0]

    comparisons = []

    ### Comparisons
    for i, contrast_name in enumerate(contrast_names):
        if preferences.use_metadata_only:
            # blocks_list = comparison_list['fields'][5][settings.comparisons][i].split(';')
            # align_to_list = comparison_list['fields'][4][settings.comparisons][i].split(';')
            blocks = comparison_list['fields'][4][i]
            align_to = comparison_list['fields'][3][i]
            generalize_to_modality = comparison_list['fields'][7][i]
            generalize_to_contrast = comparison_list['fields'][8][i]
            # for b, blocks in enumerate(blocks_list):
            #     for align_to in align_to_list:
            curr_dict = {}
            curr_dict['contrast_name'] = contrast_name + '_' + str(blocks) + '_' + align_to
            curr_dict['contrast'] = comparison_list['fields'][1][i]
            curr_query = curr_dict['contrast'][1:-1].split(',')
            curr_query = [s.strip() for s in curr_query]
            curr_dict['query'] = curr_query
            cond_labels = comparison_list['fields'][2][i]
            curr_dict['cond_labels'] = cond_labels[1:-1].split(',')
            curr_dict['align_to'] = align_to
            curr_dict['blocks'] = blocks
            curr_dict['generalize_to_blocks'] = generalize_to_modality
            curr_dict['generalize_to_contrast'] = generalize_to_contrast
            sortings = comparison_list['fields'][5][i]
            if isinstance(sortings, str):
                curr_dict['sorting'] = sortings.split(',')
            else:
                curr_dict['sorting'] = []
            curr_dict['union_or_intersection'] = comparison_list['fields'][6][i]

            comparisons.append(curr_dict)

        else:
            print('Metadata is not used')

    return comparisons


def load_POS_tags(settings):
    with open(os.path.join(settings.path2stimuli, settings.word2pos_file), 'rb') as f:
        word2pos = pickle.load(f)
        word2pos['exercised'] = word2pos['excercised']
        word2pos['stretched'] = word2pos['streched']
    return word2pos

def load_word_features(path2stimuli=os.path.join('..', '..', 'Paradigm'),
                       word_features_filename='word_features.xlsx',
                       word_features_filename_new = 'word_features_new.xlsx'):
    word2features = {}
    sheet = pd.read_excel(os.path.join(path2stimuli, word_features_filename))
    words = sheet['word_string']
    morphemes = sheet['morpheme']
    morpheme_types = sheet['morpheme_type']
    word_type = sheet['word_type'] # function or content word

    for w, m, t, cf in zip(words, morphemes, morpheme_types, word_type):
        if np.isnan(t):
            t=0
        if not isinstance(m, str):
            m=''
        word2features[w.lower()] = (m, t, cf)


    word2features['exercised'] = word2features['excercised']
    word2features['stretched'] = word2features['streched']
    
    ##
    word2features_new = {}
    sheet = pd.read_excel(os.path.join(path2stimuli, word_features_filename_new))
    sheet = sheet.loc[:, ~sheet.columns.str.contains('^Unnamed')]
    for i, row in sheet.iterrows():
        s = row['stimulus_number']
        w = row['word_position']
        #print(s, w)
        #print(row)
        if s not in word2features_new.keys():
            word2features_new[s]={}
        if w not in word2features_new[s].keys():
            word2features_new[s][w]= {}
        for f in row.keys():
            if f in ['sentence_string', 'word_string', 'pos']:
                word2features_new[s][w][f] = row[f]
            else:
                word2features_new[s][w][f] = int(row[f])


        # add for word_position=-1 (end of sentence):
        if 0 not in word2features_new[s].keys():
            word2features_new[s][0]= {}
            word2features_new[s][0]['sentence_string'] = row['sentence_string']
            for f in row.keys():
                if f in ['word_string', 'pos']:
                    word2features_new[s][0][f] = ''
                elif f == 'sentence_string':
                    pass
                else:
                    word2features_new[s][0][f] = 0

    return word2features, word2features_new


def load_comparisons_and_features(settings):
    import pandas

    # Read comparison file ('xlsx')
    sheet = pandas.read_excel(os.path.join(settings.path2stimuli, settings.comparisons_file))
    headers = sheet.columns
    fields = []
    for i, header in enumerate(headers):
        fields.append(sheet[header].values)
        comparison_list = {'headers':headers, 'fields':fields}

    del sheet, headers

    # Read features file ('xlsx')
    sheet = pandas.read_excel(os.path.join(settings.path2stimuli, settings.features_file))
    headers = sheet.columns
    fields = []
    for i, header in enumerate(headers):
        fields.append(sheet[header].values)
    features = {'headers': headers, 'fields': fields}
    
    return comparison_list, features


def get_probes2channels(patients, flag_get_channels_with_spikes=True):
    '''
    input: patient (str)
    output: probes (dict) - key is the probe names; value is a list of lists (per patient), with channel numbers for micro or macro data. For example, probes['LSTG']['micro'] = [[25, 26, ...], [36, ..]]
    '''
    def get_file_probe_names(path2mat_folder, micro_macro):
        with open(os.path.join(path2mat_folder, 'channel_numbers_to_names.txt')) as f:
            lines = f.readlines()
        channel_numbers  = [l.strip().split('\t')[0] for l in lines]
        file_names = [l.strip().split('\t')[1] for l in lines]
        if micro_macro == 'micro':
            probe_names = set([s[4:-5] for s in file_names if s.startswith('G')])
        elif micro_macro == 'macro':
            probe_names = set([s[:-5] for s in file_names])
        return channel_numbers, file_names, probe_names

    path2functions = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(path2functions)
    
    # First, find probe names from all patients
    probe_names_all_patients = []
    for patient in patients:
        # MICRO CSC
        path2microdata_folder = os.path.join(path2functions, '..', '..', '..', 'Data', 'UCLA', patient, 'Raw', 'micro', 'CSC_mat')
        channel_numbers_micro, file_names_micro, probe_names_micro = get_file_probe_names(path2microdata_folder, 'micro')
        # MACRO CSC
        path2macrodata_folder = os.path.join(path2functions, '..', '..', '..', 'Data', 'UCLA', patient, 'Raw', 'macro', 'CSC_mat')
        channel_numbers_macro, file_names_macro, probe_names_macro = get_file_probe_names(path2macrodata_folder, 'macro')
        # COMPARE micro-macro
        if not probe_names_micro == probe_names_macro:
            print('%s: --- !!! Warning: not the same probe names in micro and macro !!! ---' % patient)
            print('Micro probe names: %s' % probe_names_micro)
            print('Macro probe names: %s' % probe_names_macro)
        else:
            pass
            #print('%s: '%patient, probe_names_micro)
        # UNIFY micro-macro
        probe_names_micro_macro = list(set(list(set(probe_names_micro))+list(set(probe_names_macro))))
        probe_names_all_patients.append(probe_names_micro_macro)
    probe_names_all_patients = list(set([n for l in probe_names_all_patients for n in l]))

    # Generate a dict with channel numbers of each probe per each patient.
    probes = {}
    probes['patients'] = []
    probes['probe_names'] = {}
    for patient in patients:
        probes['patients'].append(patient)
        # CHECK CHANNELS WITH SPIKES
        settings = Settings(patient)
        if flag_get_channels_with_spikes:
            channels_with_spikes = get_channels_with_spikes_from_combinato_sorted_h5(settings, ['pos']) # TODO: fox 'neg' case
            channels_with_spikes = [sublist[0] for sublist in channels_with_spikes if (sublist[2]>0)|(sublist[3]>0)]
        else:
            channels_with_spikes = []
        print('Channels with spikes for patient %s' % patient, channels_with_spikes)
        for probe_name in probe_names_all_patients: # Take the union in case probe_names_micro != probe_names_macro
            path2microdata_folder = os.path.join(path2functions, '..', '..', '..', 'Data', 'UCLA', patient, 'Raw', 'micro', 'CSC_mat')
            channel_numbers_micro, file_names_micro, probe_names_micro = get_file_probe_names(path2microdata_folder, 'micro')
            path2macrodata_folder = os.path.join(path2functions, '..', '..', '..', 'Data', 'UCLA', patient, 'Raw', 'macro', 'CSC_mat')
            channel_numbers_macro, file_names_macro, probe_names_macro = get_file_probe_names(path2macrodata_folder, 'macro')
            channel_numbers_of_probe_micro = [int(ch) for (ch, fn) in zip(channel_numbers_micro, file_names_micro) if probe_name == fn[4:-5]]
            channel_numbers_of_probe_macro = [int(ch) for (ch, fn) in zip(channel_numbers_macro, file_names_macro) if probe_name == fn[:-5]]
            if probe_name not in probes['probe_names'].keys(): # a new probe was found - initialize patient list with channel numbers.
                probes['probe_names'][probe_name] = {}
                probes['probe_names'][probe_name]['micro'] = [] # list of lists (per patient), with which channel numbers belong to this probe
                probes['probe_names'][probe_name]['macro'] = [] # list of lists (per patient), with which channel numbers belong to this probe
                probes['probe_names'][probe_name]['spike'] = [] # list of lists (per patient), with which channel numbers have spikes
                probes['probe_names'][probe_name]['patients'] = [] # which patients have this probe
            #print(patient, probe_name, channel_numbers_of_probe_macro)
            probes['probe_names'][probe_name]['micro'].append(channel_numbers_of_probe_micro)
            probes['probe_names'][probe_name]['macro'].append(channel_numbers_of_probe_macro)
            probes['probe_names'][probe_name]['spike'].append(list(set(channels_with_spikes).intersection(channel_numbers_of_probe_micro)))
            if channel_numbers_of_probe_micro or channel_numbers_of_probe_macro:
                probes['probe_names'][probe_name]['patients'].append(patient)
    # MICROPHPNE to micro electrodes
    probes['probe_names']['MICROPHONE'] = {}
    probes['probe_names']['MICROPHONE']['micro'] = [0]



    return probes
