import os
import glob
import pickle
import sys
import numpy as np
import mne
from scipy import io
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pandas as pd
from .features import Features
from wordfreq import word_frequency, zipf_frequency
from utils.utils import probename2picks
from utils.brpylib import NsxFile, brpylib_ver
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


    def load_raw_data(self, smooth=False, decimate=False, verbose=False):
        '''

        Parameters
        ----------
        smooth : float, optional
            Smoothing window size in miliseconds. The default is None.
        verbose : TYPE, optional
            Verbosity. The default is False.

        Returns
        -------
        None.

        '''

        self.raws = []  # list of raw MNE objects
        # IXs_to_remove = []
        for p, (patient, data_type, filt) in enumerate(zip(self.patient,
                                                           self.data_type,
                                                           self.filter)):
            # Load RAW object
            path2rawdata = f'../../Data/UCLA/{patient}/Raw/mne'
            fname_raw = '%s_%s_%s-raw.fif' % (patient, data_type, filt)
            fname_raw = os.path.join(path2rawdata, fname_raw)
            
            if os.path.isfile(fname_raw):
                raw_neural = mne.io.read_raw_fif(fname_raw, preload=True)
            else:
                print('!'*100)
                print(f'Skipping file, NOT FOUND: {fname_raw}')
                print('!'*100)
                # IXs_to_remove.append(self.patient.index(patient))
                continue
            
            # PICK
            picks = None
            if self.probe_name:
                picks = probename2picks(self.probe_name[p],
                                        raw_neural.ch_names,
                                        data_type)
            if self.channel_num:
                picks = self.channel_num[p]-1
            if self.channel_name:
                picks = self.channel_name[p]
            if verbose:
                print('Trying to pick:', picks)
                print(type(picks))
            try:
                raw_neural.pick(picks)
            except Exception as errmsg:
                print('!'*100)
                print(f'PICK FAILED: {fname_raw}; {errmsg}')
                #print('All channels included')
                print(f'Skipping: {patient}, {data_type}, {filt}')
                print('!'*100)
                self.raws.append(None)
                continue
                # IXs_to_remove.append(self.patient.index(patient))
            
            # SMOOTH
            sfreq = raw_neural.info['sfreq']
            if smooth:
                width_sec = smooth/1000  # Gaussian-kernal width in [sec]
                print(f'smoothing data with {width_sec} sec window')
                data = raw_neural.copy().get_data()
                for ch in range(data.shape[0]):  # over channels
                    time_series = data[ch, :]
                    data[ch, :] = gaussian_filter1d(
                        time_series, width_sec*sfreq)
                raw_neural._data = data

            # DECIMATE
            if decimate:
                raw_neural.resample(int(raw_neural.info['sfreq']/decimate))

            # SAMPLING FREQUENCY
            self.sfreq = raw_neural.info['sfreq']

            if self.feature_list:
                metadata = prepare_metadata(patient)
                metadata = extend_metadata(metadata)
                #metadata = prepare_metadata_features(patient)
                feature_data = Features(metadata, self.feature_list)
                feature_data.add_feature_info()
                feature_data.add_design_matrix()
                #feature_data.scale_design_matrix()
                feature_data.add_raw_features(len(raw_neural), self.sfreq)

                raw_neural.load_data()
                raw_neural = raw_neural.add_channels([feature_data.raw],
                                                     force_update_info=True)

                self.feature_info = feature_data.feature_info
            self.raws.append(raw_neural)

        
        # if IXs_to_remove:
        #     for IX in IXs_to_remove:
        #         del self.patient[IX]
        #         del self.data_type[IX]
        #         del self.filter[IX]
        #         if self.channel_name:
        #             del self.channel_name[IX]
        
        if verbose:
            print(self.raws)
            [print(raw.ch_names) for raw in self.raws if raw is not None]
            print(self.patient, self.data_type, self.filter)

    def epoch_data(self, level,
                   tmin=None, tmax=None,  query=None,
                   block_type=None, scale_epochs=False, verbose=False):
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

        Returns
        -------
        None.

        '''
        print(self.patient, self.data_type, self.filter)
        self.epochs = []
        for p, (patient, data_type) in enumerate(zip(self.patient,
                                                     self.data_type)):
            print(f'Epoching {patient}, {data_type}, {level}')
            if self.raws[p] is None:
                print(f'Skipping epoching {patient} {data_type}')
                continue
            ############
            # METADATA #
            ############
            metadata = prepare_metadata(patient)
            metadata = extend_metadata(metadata)
            

            if level == 'sentence_onset':
                tmin_, tmax_ = (-1.2, 3.5)
                metadata_level = metadata.loc[((metadata['block'].isin([1, 3, 5])) &
                                              (metadata['word_position'] == 1)) |
                                              ((metadata['block'].isin([2, 4, 6])) &
                                              (metadata['word_position'] == 1) &
                                              (metadata['phone_position'] == 1))]
            elif level == 'sentence_offset':
                tmin_, tmax_ = (-3.5, 1.5)
                metadata_level = metadata.loc[(metadata['is_last_word'] == 1)]
            elif level == 'sentence_end':
                tmin_, tmax_ = (-3.5, 1.5)
                metadata_level = metadata.loc[(metadata['word_string'] == '.') | (metadata['phone_string'] == 'END_OF_WAV')]
            elif level == 'word':
                tmin_, tmax_ = (-1.5, 2.5)
                metadata_level = metadata.loc[((metadata['word_onset'] == 1) &
                                              (metadata['block'].isin([2, 4, 6]))) |
                                              ((metadata['block'].isin([1, 3, 5])) &
                                              (metadata['word_position'] > 0))] 
            elif level == 'phone':
                tmin_, tmax_ = (-0.3, 1.2)
                metadata_level = metadata.loc[(metadata['block'].isin([2, 4, 6])) &
                                              (metadata['phone_position'] > 0)]
            elif level == 'block':
                metadata_level = metadata.loc[(metadata['chronological_order'] == 1) |
                                              (metadata['chronological_order'] == 509) |
                                              (metadata['chronological_order'] == 2127) |
                                              (metadata['chronological_order'] == 2635) |
                                              (metadata['chronological_order'] == 4253) |
                                              (metadata['chronological_order'] == 4761)]
            else:
                raise('Unknown level type (sentence_onset/sentence_offset/word/phone)')
            
            
            ##########
            # EVENTS #
            ##########
            events, event_id = get_events(metadata_level, self.sfreq)
            
            if verbose:
                if self.raws[p] is not None:
                    print(self.raws[p].first_samp, events)
            
            ############
            # EPOCHING #
            ############
            epochs = mne.Epochs(self.raws[p], events, event_id, tmin_, tmax_,
                                metadata=metadata_level,
                                baseline=None,
                                reject=None,
                                preload=True)
            if any(epochs.drop_log):
                print('Dropped:', epochs.drop_log)

            if block_type == 'auditory':
                epochs = epochs['block in [2, 4, 6]']
            elif block_type == 'visual':
                epochs = epochs['block in [1, 3, 5]']
            # QUERY
            if query:
                epochs = epochs[query]
            if verbose:
                print(query)
                print(epochs)
            # CROP
            if tmin and tmax:
                epochs = epochs.crop(tmin=tmin, tmax=tmax)
            

            # Separate neural data from features before pick and scale
            epochs_neural = epochs.copy().pick_types(seeg=True, eeg=True)
            if self.feature_list:
                epochs_features = epochs.copy().pick_types(misc=True)


            #if smooth:
            #    width_sec = smooth/1000  # Gaussian-kernal width in [sec]
            #    print(f'smoothing data with {width_sec} sec window')
            #    data = epochs_neural.copy().get_data()
            #    for ch in range(data.shape[1]):  # over channels
            #        for tr in range(data.shape[0]):  # over trials
            #            time_series = data[tr, ch, :]
            #            data[tr, ch, :] = gaussian_filter1d(
            #                time_series, width_sec*self.sfreq)
            #    epochs_neural._data = data
            
            
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


def get_events(metadata, sfreq, verbose=False):

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
    return events, event_id


def generate_mne_raw(data_type, from_mat, path2rawdata, sfreq_down, ch_names_from_file):
    
    assert not (data_type == 'spike' and from_mat)
    
    # Path to data
    path2data = os.path.join(path2rawdata, data_type)
    if from_mat:
        path2data = os.path.join(path2data, 'mat')
    print(f'Loading data from: {path2data}')
    
    # Extract raw data
    if data_type == 'spike':
        channel_data, sfreq, ch_names = get_data_from_combinato(path2data)
    else:
        if from_mat:
            channel_data, sfreq, ch_names = get_data_from_mat(data_type, path2data, ch_names_from_file)
        else:
            raw = get_data_from_ncs_or_ns(data_type, path2data, sfreq_down)
            return raw
    #print(f'Shape channel_data: {channel_data.shape}')
    n_channels = channel_data.shape[0]
    ch_types = ['seeg'] * n_channels
    #print(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(channel_data, info)

    del channel_data 
    return raw


def get_data_from_combinato(path2data):
    
    patient = path2data.split('patient_')[1][:3]

    sfreq = 1000

    print(f'Loading spike cluster data from: {path2data}')
    
    CSC_folders = glob.glob(os.path.join(path2data, 'CSC?/')) + \
                  glob.glob(os.path.join(path2data, 'CSC??/')) + \
                  glob.glob(os.path.join(path2data, 'CSC???/'))
    
    assert len(CSC_folders) > 0
    CSC_nums = [int(os.path.basename(os.path.normpath(s))[3::]) for s in CSC_folders]
    IX = np.argsort(CSC_nums)
    CSC_folders = np.asarray(CSC_folders)[IX]
    
    recording_system = identify_recording_system(path2data)
    if recording_system == 'Neuralynx':
        reader = neo.io.NeuralynxIO(os.path.join(path2data, '..', 'micro'))        
        channel_tuples = reader.header['signal_channels']
        time0, timeend = reader.global_t_start, reader.global_t_stop    
    elif recording_system == 'BlackRock':
        #fn_br = glob.glob(os.path.join(path2data, '..', 'micro', '*.ns5'))
        #assert len(fn_br) == 1
        #nsx_file = NsxFile(fn_br[0])
        time0, timeend = 0, 5400 # set timeend to max of 1.5 hours.
        # Extract data - note:
        # Data will be returned based on *SORTED* elec_ids,
        # see cont_data['elec_ids']
        #cont_data = nsx_file.getdata()
        #nsx_file.close()
        #time0, timeend = cont_data['start_time_s'], cont_data['data_time_s']

    print(f'time0 = {time0}, timeend = {timeend}')
    
    ch_names, spike_times_samples = [], []
    for i_ch, CSC_folder in enumerate(CSC_folders):
        CSC_num = int(CSC_folder.split('CSC')[-1].strip('/'))
        spikes, curr_ch_names = load_combinato_sorted_h5(path2data, CSC_num, i_ch)
        assert len(curr_ch_names)==len(spikes)

        if len(spikes) > 0:
            print(CSC_num, curr_ch_names)
            ch_names.extend(curr_ch_names)
            for groups, curr_spike_times_msec in enumerate(spikes):
                curr_spike_times_samples = [int(t*sfreq/1e3) for t in curr_spike_times_msec] # convert to samples from sec
                spike_times_samples.append(curr_spike_times_samples)
        else:
            print(f'No spikes in channel: {CSC_num}')

    # ADD to array
    num_groups = len(spike_times_samples)
    channel_data = np.zeros((num_groups, int(1e3*(timeend- time0 + 1))))
    for i_st, st in enumerate(spike_times_samples):
        st = (np.asarray(st) - time0*1e3).astype(int)
        channel_data[i_st, st] = 1
    
    return channel_data, sfreq, ch_names


def get_data_from_mat(data_type, path2data, ch_names_from_file):
     
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
        if 'sr' in curr_channel_data.keys():
            sfreq = curr_channel_data['sr']
        else:
            sfreq = int(1/curr_channel_data['samplingInterval'])
        channel_data.append(curr_channel_data['data'])
        
        if ch_names_from_file:
            dict_ch_names, _ = get_dict_ch_names(os.path.join(path2data, '..'))            
            ch_name = dict_ch_names[i_ch+1]
        else: 
            if 'channelName' in curr_channel_data:
                ch_name = curr_channel_data['channelName']
            else:
                ch_name = os.path.basename(CSC_file)[:-4]
        ch_names.append(ch_name)
        print(f'Processing file: {ch_name} ({i_ch+1}/{len(CSC_files)}), sfreq = {sfreq} Hz')
    channel_data = np.vstack(channel_data)

    return channel_data, sfreq, ch_names


def get_data_from_ncs_or_ns(data_type, path2data, sfreq_down):
    print(path2data, os.getcwd())
    if data_type == 'microphone':
        # Assumes that if data_type is microphone 
        # Then the recording system is Neurlanyx.
        # Otherwise, The flag --from-mat should be used
        recording_system = 'Neuralynx'
    else:
        recording_system = identify_recording_system(path2data)
    print(f'Recording system identified - {recording_system}')
    
    if recording_system == 'Neuralynx':
        reader = neo.rawio.NeuralynxRawIO(dirname=path2data)
        reader.parse_header()
        print(dir(reader))
        time0, timeend = reader.global_t_start, reader.global_t_stop
        sfreq = reader.get_signal_sampling_rate()
        print(f'Neural files: Start time {time0}, End time {timeend}')
        print(f'Sampling rate [Hz]: {sfreq}')
        channels = reader.header['signal_channels']
        ch_names = [channel[0] for channel in channels]
        print('Number of channel %i: %s'
                      % (len(ch_names), ch_names))

        n_segments_list = reader.header['nb_segment'] # list of n_segments
        chunks = []
        for i_block, n_segments in enumerate(n_segments_list):
            for i_segment in range(n_segments):
                if i_segment % 100 == 0:
                    print(f'block {i_block+1}; Segment {i_segment+1}/{n_segments}')
                curr_chunk = reader.get_analogsignal_chunk(block_index=i_block,
                                                           seg_index=i_segment)
                chunks.append(curr_chunk)
        data = np.vstack(chunks)
        info = mne.create_info(ch_names=ch_names,
                               sfreq=sfreq, ch_types='seeg')
        raw = mne.io.RawArray(data.T, info, verbose=True)
        print(raw)
        
    elif recording_system == 'BlackRock':
        if data_type == 'macro':
            ext = 'ns3'
        elif data_type == 'micro':
            ext = 'ns5'
        fn_br = glob.glob(os.path.join(path2data, '*.' + ext))
        assert len(fn_br) == 1
        nsx_file = NsxFile(fn_br[0])
        # Extract data - note:
        # Data will be returned based on *SORTED* elec_ids,
        # see cont_data['elec_ids']
        cont_data = nsx_file.getdata()
        print(cont_data)
        nsx_file.close()
        sfreq = cont_data['samp_per_s']
        print(sfreq)
        elec_ids = np.asarray(cont_data['elec_ids'])
        
        dict_ch_names, _ = get_dict_ch_names(os.path.join(path2data))
        ch_names = []
        if data_type == 'macro':
            elec_ids -= 128
        elif data_type == 'micro':
            elec_ids = np.asarray(list(set(elec_ids) - set(range(129, 300))))
            
        print(elec_ids)
        for elec_id in elec_ids:
            if data_type == 'macro':
                curr_elec_id = int(elec_id) - 128
            else:
                curr_elec_id = int(elec_id)
            
            # RETRIEVE CHANNEL NAME FROM TXT FILE
            if curr_elec_id in dict_ch_names.keys():
                ch_name = dict_ch_names[curr_elec_id]
            else:
                ch_name = f'CSC{elec_id}'
            ch_names.append(ch_name)
        #print(ch_names)
        
        # TO MNE
        info = mne.create_info(ch_names=ch_names,
                               sfreq=sfreq,
                               ch_types='seeg')
        raw = mne.io.RawArray(cont_data['data'][0].T, info, verbose=False)
    return raw


def ns2mat(data_type, path2data):
    if data_type == 'macro':
        ext = 'ns3'
    elif data_type == 'micro':
        ext = 'ns5'
    fn_br = glob.glob(os.path.join(path2data, '*.' + ext))
    assert len(fn_br) == 1
    
    # EXTRACT NEURAL DATA
    nsx_file = NsxFile(fn_br[0])
    print(dir(nsx_file))
    print(nsx_file.basic_header)
    cont_data = nsx_file.getdata()
    nsx_file.close()
    print(cont_data.keys()) 
    return cont_data['data'], cont_data['elec_ids'], cont_data['ExtendedHeaderIndices'], cont_data['samp_per_s']


def identify_recording_system(path2data):
    print(os.getcwd())
    print(path2data)
    
    # get recording system for spiking data based on micro
    if os.path.dirname(path2data) == 'spike': 
        path2data = os.path.join(os.path.dirname(path2data), 'micro')
    
    if len(glob.glob(os.path.join(path2data, '*.ncs'))):
        recording_system = 'Neuralynx'
        # assert neural_files[0][-3:] == 'ncs'
    elif len(glob.glob(os.path.join(path2data, '*.ns*')))>0 or len(glob.glob(os.path.join(path2data, '*.mat')))>0:
        recording_system = 'BlackRock'
    else:
        print(f'No neural files found: {path2data}')
        raise()

    return recording_system


def load_combinato_sorted_h5(path2data, channel_num, i_ch):  # , probe_name):
    target_types = [2] # -1: artifact, 0: unassigned, 1: MU, 2: SU

    dict_ch_names, _ = get_dict_ch_names(os.path.join(path2data, '..', 'micro'))
    ch_name = dict_ch_names[i_ch+1]

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
                        print(f'found cluster in {ch_name}, channel {channel_num}, group {g}')
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
                        #if isinstance(probe_name, list):
                        #    if probe_name:
                        #        pn=probe_name[0]
                        #    else:
                        #        pn=None
                        #elif isinstance(probe_name, str):
                        #    pn=probe_name
                        group_names.append(f"{ch_name.split('-')[1]}_{i_ch}_{sign[0]}{g}")
            else:
                print('%s was not found!' % os.path.join(path2data, 'micro', 'CSC' + str(channel_num), 'sort_' + sign + '_yl2', 'sort_cat.h5'))

    else:
        print('None or more than a single combinato h5 was found')

    return spike_times_msec, group_names





def add_event_to_metadata(metadata, event_time, sentence_number, sentence_string, word_position, word_string, pos, num_words, last_word):
    metadata['event_time'].append(event_time)
    metadata['sentence_number'].append(sentence_number)
    metadata['sentence_string'].append(sentence_string.capitalize())
    metadata['word_position'].append(word_position)
    metadata['word_string'].append(word_string)
    metadata['pos'].append(pos)
    metadata['num_words'].append(num_words)
    metadata['last_word'].append(last_word)
    return metadata



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
        # 864156059 PHONE_ONSET 1 0 1 1 1 HH HE
        lines = [l for l in lines if l[1]=='PHONE_ONSET']
        events['event_time'] = [l[0] for l in lines]
        events['block'] = len(events['event_time']) * [block]
        events['word_onset'] = [1 if int(l[2])>0 else 0 for l in lines]
        events['is_last_phone'] = [int(l[3]) for l in lines]
        events['phone_position'] = [int(l[4]) for l in lines]
        events['phone_string'] = [l[7] for l in lines]
        events['word_position'] = [int(l[5]) for l in lines]
        events['word_string'] = [l[8] for l in lines]
        events['stimulus_number'] = [int(l[6]) for l in lines]

    elif block in [1, 3, 5]:
        lines = [l for l in lines if l[1] == 'DISPLAY_TEXT' and l[2] != 'OFF']
        events['event_time'] = [l[0] for l in lines]
        events['block'] = len(events['event_time']) * [block]
        events['phone_position'] = len(events['event_time']) * [0] # not relevant for visual blocks
        events['phone_string'] = len(events['event_time']) * ['']  # not relevant for visual blocks
        events['is_last_phone'] = len(events['event_time']) * ['']  # not relevant for visual blocks
        events['word_position'] = [int(l[4]) for l in lines]
        events['word_string'] = [l[5] for l in lines]
        events['stimulus_number'] = [int(l[3]) for l in lines]
        events['word_onset'] = [1 if len(ws)>1 else 0 for ws in events['word_string']]
    return events


def prepare_metadata(patient, verbose=False):
    '''
    :param log_all_blocks: list len = #blocks
    :param features: numpy
    :return: metadata: list
    '''
    blocks = range(1, 7)

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

    word_ON_duration = 200 # [msec]
    word2features, word2features_new = load_word_features()
    n_categories = len(word2features['semantic_categories'])
    #print(word2features['semantic_categories'])
    #print(word2features_new)
    num_blocks = len(log_all_blocks)

    # Create a dict with the following keys:
    keys = ['chronological_order', 'event_time', 'block', 'word_onset', 'phone_position', 'phone_string', 'is_last_phone', 'stimulus_number',
            'word_position', 'word_string', 'pos', 'dec_quest', 'grammatical_number', 'wh_subj_obj',
            'word_length', 'sentence_string', 'sentence_length', 'last_word', 'morpheme', 'morpheme_type', 'word_type', 'word_freq', 'word_zipf',
            'gender', 'n_open_nodes', 'tense', 'syntactic_role', 'diff_thematic_role', 'unaccusative', 'semantic_categories', 'semantic_categories_names']
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
            metadata['event_time'].append(int(float(curr_block_events['event_time'][i])) / 1e6)
            metadata['block'].append(curr_block_events['block'][i])
            word_onset = curr_block_events['word_onset'][i]
            if word_onset==-1: word_onset=0
            metadata['word_onset'].append(word_onset)
            phone_pos = int(curr_block_events['phone_position'][i])
            metadata['phone_position'].append(phone_pos)
            
            is_last_phone = curr_block_events['is_last_phone'][i]
            metadata['is_last_phone'].append(is_last_phone)
            
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
            middle_word_onset = wp!=1 and ((curr_block_events['phone_string'][i] != 'END_OF_WAV' and phone_pos>1 and word_onset) or (curr_block_events['block'][i] in [1,3,5]))
            middle_phone = (curr_block_events['phone_string'][i] != 'END_OF_WAV' and (not word_onset) and (curr_block_events['block'][i] in [2,4,6]))
            if sentence_onset: # ADD WORD AND- SENTENCE-LEVEL FEATURES
                metadata['sentence_string'].append(word2features_new[sn][wp]['sentence_string'])
                metadata['sentence_length'].append(word2features_new[sn][wp]['sentence_length'])
                metadata['word_length'].append(word2features_new[sn][wp]['word_length'])
                metadata['dec_quest'].append(word2features_new[sn][wp]['dec_quest'])
                metadata['gender'].append(word2features_new[sn][wp]['gender'])
                metadata['n_open_nodes'].append(word2features_new[sn][wp]['n_open_nodes'])
                metadata['tense'].append(word2features_new[sn][wp]['tense'])
                metadata['syntactic_role'].append(word2features_new[sn][wp]['syntactic_role'])
                metadata['diff_thematic_role'].append(word2features_new[sn][wp]['diff_thematic_role'])
                metadata['unaccusative'].append(word2features_new[sn][wp]['unaccusative'])
                metadata['grammatical_number'].append(word2features_new[sn][wp]['grammatical_number'])
                metadata['pos'].append(word2features_new[sn][wp]['pos'])
                metadata['wh_subj_obj'].append(word2features_new[sn][wp]['wh_subj_obj'])
                metadata['morpheme'].append(word2features[word_string][0])
                metadata['morpheme_type'].append(int(word2features[word_string][1]))
                metadata['word_type'].append(word2features[word_string][2])
                metadata['semantic_categories'].append(word2features[word_string][3])
                metadata['semantic_categories_names'].append(word2features[word_string][4])
                metadata['last_word'].append(metadata['sentence_length'][-1] == metadata['word_position'][-1])
                metadata['word_freq'].append(word_freq)
                metadata['word_zipf'].append(word_zipf)
            elif middle_word_onset: # ADD WORD-LEVEL FEATURES
                metadata['sentence_string'].append(word2features_new[sn][wp]['sentence_string'])
                metadata['sentence_length'].append(word2features_new[sn][wp]['sentence_length'])
                metadata['word_length'].append(word2features_new[sn][wp]['word_length'])
                metadata['dec_quest'].append(word2features_new[sn][wp]['dec_quest'])
                metadata['gender'].append(word2features_new[sn][wp]['gender'])
                metadata['n_open_nodes'].append(word2features_new[sn][wp]['n_open_nodes'])
                metadata['tense'].append(word2features_new[sn][wp]['tense'])
                metadata['syntactic_role'].append(word2features_new[sn][wp]['syntactic_role'])
                metadata['diff_thematic_role'].append(word2features_new[sn][wp]['diff_thematic_role'])
                metadata['unaccusative'].append(word2features_new[sn][wp]['unaccusative'])
                metadata['grammatical_number'].append(word2features_new[sn][wp]['grammatical_number'])
                metadata['pos'].append(word2features_new[sn][wp]['pos'])
                metadata['wh_subj_obj'].append(0)
                metadata['morpheme'].append(word2features[word_string][0])
                metadata['morpheme_type'].append(int(word2features[word_string][1]))
                metadata['word_type'].append(word2features[word_string][2])
                metadata['semantic_categories'].append(word2features[word_string][3])
                metadata['semantic_categories_names'].append(word2features[word_string][4])
                metadata['last_word'].append(metadata['sentence_length'][-1] == metadata['word_position'][-1])
                metadata['word_freq'].append(word_freq)
                metadata['word_zipf'].append(word_zipf)
            elif middle_phone:  # NO WORD/SENTENCE-LEVEL FEATURES
                metadata['sentence_string'].append(
                    word2features_new[sn][wp]['sentence_string'])
                metadata['sentence_length'].append(
                    word2features_new[sn][wp]['sentence_length'])
                metadata['word_length'].append(0)
                metadata['dec_quest'].append(word2features_new[sn][wp]['dec_quest'])
                metadata['gender'].append(0)
                metadata['n_open_nodes'].append(0)
                metadata['tense'].append(0)
                metadata['syntactic_role'].append(0)
                metadata['diff_thematic_role'].append(0)
                metadata['unaccusative'].append(0)
                metadata['grammatical_number'].append(0)
                metadata['pos'].append('')
                metadata['wh_subj_obj'].append(0)
                metadata['morpheme'].append('')
                metadata['morpheme_type'].append('')
                metadata['word_type'].append('')
                metadata['semantic_categories'].append(np.zeros(n_categories))
                metadata['semantic_categories_names'].append(np.zeros(n_categories))
                metadata['last_word'].append(False)
                metadata['word_freq'].append(0)
                metadata['word_zipf'].append(0)
            elif curr_block_events['phone_string'][i] == 'END_OF_WAV':
                metadata['sentence_string'].append(
                    word2features_new[sn][wp]['sentence_string'])
                metadata['sentence_length'].append(
                    word2features_new[sn][wp]['sentence_length'])
                metadata['word_length'].append(0)
                metadata['dec_quest'].append(word2features_new[sn][1]['dec_quest'])
                metadata['gender'].append(0)
                metadata['n_open_nodes'].append(0)
                metadata['tense'].append(0)
                metadata['syntactic_role'].append(0)
                metadata['unaccusative'].append(0)
                metadata['diff_thematic_role'].append(0)
                metadata['grammatical_number'].append(0)
                metadata['pos'].append('')
                metadata['wh_subj_obj'].append(0)
                metadata['morpheme'].append('')
                metadata['morpheme_type'].append('')
                metadata['word_type'].append('')
                metadata['semantic_categories'].append(np.zeros(n_categories))
                metadata['semantic_categories_names'].append(np.zeros(n_categories))
                metadata['last_word'].append(False)
                metadata['word_freq'].append(0)
                metadata['word_zipf'].append(0)
                metadata['phone_position'][-1] = 0
                metadata['is_last_phone'][-1] = 0
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
                metadata['word_onset'].append(0)
                metadata['phone_position'].append(0)
                metadata['is_last_phone'].append(0)
                metadata['phone_string'].append('')
                metadata['stimulus_number'].append(
                    int(curr_block_events['stimulus_number'][i]))
                metadata['word_position'].append(0)
                metadata['word_string'].append('.')
                metadata['pos'].append('')
                metadata['morpheme'].append('')
                metadata['morpheme_type'].append('')
                metadata['word_type'].append('')
                metadata['semantic_categories'].append(np.zeros(n_categories))
                metadata['semantic_categories_names'].append(np.zeros(n_categories))
                metadata['word_freq'].append(0)
                metadata['word_zipf'].append(0)
                metadata['sentence_string'].append(
                    word2features_new[sn][wp]['sentence_string'])
                metadata['sentence_length'].append(
                    word2features_new[sn][wp]['sentence_length'])
                metadata['word_length'].append(0)
                metadata['dec_quest'].append(word2features_new[sn][wp]['dec_quest'])
                metadata['gender'].append(0)
                metadata['n_open_nodes'].append(0)
                metadata['tense'].append(0)
                metadata['syntactic_role'].append(0)
                metadata['diff_thematic_role'].append(0)
                metadata['unaccusative'].append(0)
                metadata['grammatical_number'].append(0)
                metadata['wh_subj_obj'].append(0)
                metadata['last_word'].append(False)

    metadata = pd.DataFrame(metadata)
    metadata.sort_values(by='event_time')
    return metadata


def block_type(row):
    if row['block'] in [1,3,5]:
        block_type = 'visual'
    elif row['block'] in [2,4,6]:
        block_type = 'auditory'
    else:
        block_type = 'other'
    return block_type


def extend_metadata(metadata):
    ''' Add columns to metadata
    '''
    metadata = metadata.rename(columns={'last_word': 'is_last_word'})
     # FIX ORTHOGRAPHIC MISTAKES
    metadata['word_string'] = metadata['word_string'].replace({'excercised':'exercised', 'heared':'heard', 'streched':'stretched'})
   

    metadata['block_type'] = metadata.apply(lambda row: block_type(row), axis=1)
    metadata['first_word'] = metadata.apply(lambda row: row['sentence_string'].split()[0].capitalize(), axis=1)
    metadata['second_word'] = metadata.apply(lambda row: row['sentence_string'].split()[1].lower(), axis=1)
    metadata['word_position_reversed'] = metadata.apply(lambda row: 1 + row['sentence_length'] - row['word_position'], axis=1)
    # TENSE
    # LAST LETTER OF POS OF VERBS INDICATE THE TENSE (D - past, P - present, F - future, V - passive, I - infinitive-like, G - ing)
    poss = metadata['pos']
    #tenses = []
    #dict_tense = {'D':'past', 'P':'present', 'F':'future', 'V':'passive', 'I':'inf_like', 'G':'ing'}
    #for i_pos, pos in enumerate(poss):
        # print(i_pos, pos)
    #    if pos.startswith('VB'):
    #        tense = dict_tense[pos[-1]]
    #        if tense == 'passive': tense = 'past' # HACK: all passive forms are in past
    #    else: # not a verb
    #        tense = ''
    #    tenses.append(tense)
    #metadata['tense'] = tenses
    
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
                          metadata['word_onset']):
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
    for curr_sn, curr_wp, curr_ws in zip(metadata['stimulus_number'], metadata['word_position'], metadata['word_string']):
        # same stimulus/sentence number (sn) but great word-position (wp)
        is_in_embedding = any([1 if (curr_sn == sn and (curr_wp>=wp or curr_wp==0)) else 0 for (sn, wp) in stim_numbers_with_that])
        #print(curr_sn, curr_wp, curr_ws, is_in_embedding)
        embedding.append(is_in_embedding)
    metadata['embedding'] = embedding

    # WH-subj_obj
    dict_sn2wh = {}
    for i, row in metadata.iterrows():
        wh = row['wh_subj_obj']
        if wh in [-1, 1]:
            sn = row['stimulus_number']
            dict_sn2wh[sn] = wh

    for i, row in metadata.iterrows():
        if row['stimulus_number'] in dict_sn2wh.keys() and row['word_onset']:
            metadata.loc[i, 'wh_subj_obj'] = dict_sn2wh[row['stimulus_number']]

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
    metadata['glove'] = X            
    
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
    
    metadata['letter_by_position'] = get_letter_by_position(metadata)
    metadata['phone_by_position'] = get_phone_by_position(metadata)
    
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
    semantic_features = sheet['semantic_feature']

    # k-hot representation of semantic features
    n_words = len(semantic_features)
    semantic_features = [list(map(lambda s:s.strip(), l.split(','))) for l in semantic_features] # list of sublists (of semantic features)
    semantic_features_unique = sorted(list(set().union(*semantic_features))) # unique set of semantic features
    n_semantic_features = len(semantic_features_unique)
    semantic_features_IXs = [[semantic_features_unique.index(e) for e in sl] for sl in semantic_features] # list of sublists of indexes
    semantic_features_k_hot = np.zeros((n_words, n_semantic_features))
    for i_word, IXs_features in enumerate(semantic_features_IXs):
        semantic_features_k_hot[i_word, IXs_features] = 1
    word2features['semantic_categories'] = semantic_features_unique
    #print(semantic_features_unique)

    for w, m, t, cf, sem_feat_khot, sem_feat in zip(words, morphemes, morpheme_types, word_type, semantic_features_k_hot, semantic_features):
        if np.isnan(t):
            t=0
        if not isinstance(m, str):
            m=''
        word2features[w.lower()] = (m, t, cf, sem_feat_khot, ",".join(sem_feat))


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
        #print(row.keys())
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
                elif f == 'sentence_length':
                    word2features_new[s][0][f] = len(row['sentence_string'].split())
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
        with open(os.path.join(path2mat_folder, 'channel_numbers_to_names.txt'), 'r') as f:
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


# def decimate_with_feature_preserving(raw_features, decimate, sfreq_new):
#     '''
#     Decimation that makes sure that non-zero values are not get lost.

#     Parameters
#     ----------
#     raw_features : MNE raw object
#         DESCRIPTION.
#     decimate : int
#         DESCRIPTION.

#     Returns
#     -------
#     raw_features : MNE raw object
#         DESCRIPTION.

#     '''
#     # prepare new data array
#     n_channels, n_times = raw_features._data.shape
#     n_times_new = np.ceil(n_times/decimate)
#     data_decimated = np.zeros_like(n_channels, n_times_new,)
#     # find indices of non-zero values and devide them by decimate factor
#     IXs = list(np.nonzero(raw_features._data))
#     values = raw_features._data[IXs]
#     IXs[1] = np.round(IXs[1]/decimate).astype(int)
#     IXs = tuple(IXs)
#     # update new data array
#     data_decimated[IXs] = values
#     raw_features._data = data_decimated
#     # update sfreq of feature data,
#     # based on neural data after mne standard decimation
#     raw_features.info['sfreq'] = sfreq_new

#     return raw_features


def get_dict_ch_names(path2data):
    fn_channel_names = os.path.join(path2data, 'channel_numbers_to_names.txt')
    with open(fn_channel_names, 'r') as f:
        lines = f.readlines()
    keys = [int(ll.split()[0]) for ll in lines]
    values = [ll.split()[1] for ll in lines]
    
    return dict(zip(keys, values)), values


def get_letter_by_position(df_metadata):
    alphabet=[letter for letter in 'abcdefghijklmnopqrstuvwxyz']
    n_letters = len(alphabet)
    
    word_strings = df_metadata['word_string']
    letter_by_position = []
    for word_string in word_strings:
        encoding_vec = np.zeros((1, n_letters*3))
        for i_letter, letter in enumerate(word_string):
            if letter.lower() in alphabet:
                if i_letter == 0: # First letter
                    i_pos = 0
                elif i_letter == len(word_string) - 1: # Last letter 
                    i_pos = 2
                else: # middle letter
                    i_pos = 1
                IX = i_pos * n_letters + alphabet.index(letter.lower())
                encoding_vec[0, IX] = 1
        letter_by_position.append(encoding_vec)
   
    return letter_by_position
    

def get_phone_by_position(df_metadata):
    phone_set = ['AA1', 'AA2',  'AE1', 'AH0', 'AH1', 'AO1', 'AW1', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH1', 'END_OF_WAV', 'ER0', 'ER1',
                 'EY1', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW1', 'OW2', 'OY1', 'P', 'R',
                  'S', 'SH', 'T', 'TH', 'UH1', 'UW0', 'UW1', 'V', 'W', 'Z']
    n_phones = len(phone_set)

    phones = df_metadata['phone_string']
    poss = df_metadata['phone_position']
    is_last_phones = df_metadata['is_last_phone']
    is_first_phones = df_metadata['word_onset']
    
    phone_by_position = []
    for phone, pos, is_first_phone, is_last_phone in zip(phones, poss, is_first_phones, is_last_phones):
        encoding_vec = np.zeros((1, n_phones*3))
        if phone not in phone_set:
            phone_by_position.append(encoding_vec)
        else:
            if is_first_phone:
                i_pos = 0
            elif is_last_phone:
                i_pos = 2
            else:
                i_pos = 1

            i_phone = phone_set.index(phone)
            IX = i_pos * n_phones + i_phone
            encoding_vec[0, IX] = 1
            phone_by_position.append(encoding_vec)

    return phone_by_position



