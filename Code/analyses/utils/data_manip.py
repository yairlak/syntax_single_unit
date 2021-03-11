import os, glob, sys
import numpy as np
import mne
from scipy import io

def get_channel_nums(path2rawdata):
    CSC_files = glob.glob(os.path.join(path2rawdata, 'micro', 'CSC_mat', 'CSC?.mat')) + \
                glob.glob(os.path.join(path2rawdata, 'micro', 'CSC_mat', 'CSC??.mat')) + \
                glob.glob(os.path.join(path2rawdata, 'micro', 'CSC_mat', 'CSC???.mat'))
    return [int(os.path.basename(s)[3:-4]) for s in CSC_files]


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
    import load_settings_params
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
        settings = load_settings_params.Settings(patient)
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


def load_channel_data(data_type, filt, channel_num, channel_name, probe_name, settings, params):
    ''' Generate mne raw object from channel number(s), for either micro/macro/spike data.
        input -
        channel_nums: (list for macro, int for micro) channel numbers
        return -
        MNE raw object with all channels
    '''
    if data_type == 'micro' or  data_type == 'macro':
        print('Loading %s CSC data' % data_type.upper())
        channel_data = load_CSC_file(settings.path2rawdata, data_type, filt, channel_num)
        if channel_num == 0: #MICROPHONE
            ch_type = 'misc'
        else:
            #ch_type = 'seeg' if data_type == 'micro' else 'ecg'
            ch_type = 'seeg'
        #if filt == 'high-gamma':
        #    sfreq = 1000;
        #else:
        if data_type == 'micro':
            sfreq = params.sfreq_raw
        elif data_type == 'macro':
            sfreq = params.sfreq_macro

        info = mne.create_info(ch_names=[channel_name], sfreq=sfreq, ch_types=[ch_type])
        raw = mne.io.RawArray(channel_data, info)
    elif data_type == 'spike':
        print('Loading spike cluster data')
        spikes, group_names = load_combinato_sorted_h5(settings, channel_num, probe_name)
        #[print(np.max(s)) for s in spikes]
        if len(spikes) > 0:
            time0_sec = settings.time0 / 1e6
            sfreq = params.sfreq_spikes
            num_groups = len(spikes)
            ch_types = ['eeg' for _ in range(num_groups)]
            info = mne.create_info(ch_names=group_names, sfreq=sfreq, ch_types=ch_types)

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
        else:
            print('No spikes in channel:', channel_num)
            raw = None
    return raw


def load_CSC_file(path2rawdata, data_type, filt, channel_num):
    #filt_str = ''
    #if filt == 'high-gamma':
    #    filt_str = 'HighGamma_'
    CSC_file = glob.glob(os.path.join(path2rawdata, data_type, 'CSC_mat', 'CSC' + str(channel_num) + '.mat'))
    print(CSC_file)
    assert len(CSC_file)==1
    print(io.loadmat(CSC_file[0]))
    channel_data = io.loadmat(CSC_file[0])['data']
    return channel_data


def load_combinato_sorted_h5(settings, channel_num, probe_name):
    import h5py
    spike_times_msec = []; group_names = []
    if settings.time0 == 0: # BlackRock
        h5_folder = 'CSC_mat' # since combinato clusters based on mat files
    else:
        h5_folder = 'CSC_ncs' # Neuralynx case
    h5_files = glob.glob(os.path.join(settings.path2rawdata, 'micro', h5_folder, 'CSC' + str(channel_num), 'data_*.h5'))
    if len(h5_files) == 1:
        filename = h5_files[0]
        f_all_spikes = h5py.File(filename, 'r')

        for sign in ['neg', 'pos']:
        #for sign in ['neg']:
            #filename_sorted = glob.glob(os.path.join(settings.path2rawdata, 'micro', 'CSC_ncs', 'CSC' + str(channel_num), 'sort_' + sign + '_simple', 'sort_cat.h5'))[0]
            filename_sorted = glob.glob(os.path.join(settings.path2rawdata, 'micro', h5_folder, 'CSC' + str(channel_num), 'sort_' + sign + '_yl2', 'sort_cat.h5'))
            if len(filename_sorted) == 1:
                f_sort_cat = h5py.File(filename_sorted[0], 'r')
                
                #print('classes', f_sort_cat['classes'].value)
                #print('index', f_sort_cat['index'].value)
                #print('matches', f_sort_cat['matches'].value)
                #print('groups', f_sort_cat['groups'].value)
                #print('types', f_sort_cat['types'].value)
                try:
                    classes =  f_sort_cat['classes'].value
                    index = f_sort_cat['index'].value
                    matches = f_sort_cat['matches'].value
                    groups = f_sort_cat['groups'].value
                    group_numbers = set([g[1] for g in groups])
                    types = f_sort_cat['types'].value # -1: artifact, 0: unassigned, 1: MU, 2: SU

                    # For each group, generate a list with all spike times and append to spike_times
                    for g in list(group_numbers):
                        IXs = []
                        type_of_curr_group = [t_ for (g_, t_) in types if g_ == g]
                        if len(type_of_curr_group) == 1:
                            type_of_curr_group = type_of_curr_group[0]
                        else:
                            raise ('issue with types: more than one group assigned to a type')
                        if type_of_curr_group>0: # ignore artifact and unassigned groups
                            print('found cluster')
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

                            curr_spike_times = f_all_spikes[sign]['times'].value[IXs]
                            spike_times_msec.append(curr_spike_times)
                            #region_name = channel_name[1+channel_name.find("-"):channel_name.find(".")]
                            group_names.append(sign[0] + '_g' + str(g) + '_' + str(channel_num)+ '_' + probe_name)
                except:
                    print('Something is wrong with %s, %s' % (sign, filename_sorted[0]))
            else:
                print('%s was not found!' % os.path.join(settings.path2rawdata, 'micro', 'CSC_ncs', 'CSC' + str(channel_num), 'sort_' + sign + '_yl2', 'sort_cat.h5'))

        #print(channel_num)
        #print(channel_names)

    else:
        print('None or more than a single combinato h5 was found')

    return spike_times_msec, group_names


def get_channels_with_spikes_from_combinato_sorted_h5(settings, signs):
    import h5py
    # GET ALL CHANNELS NAMES FOR CURRENT SUBJECT
    path2functions = os.path.dirname(os.path.abspath(__file__))
    settings.path2rawdata = os.path.join(path2functions, '..', '..', '..', 'Data', 'UCLA', settings.patient, 'Raw')
    path2CSC_mat = os.path.join(settings.path2rawdata, 'micro', 'CSC_mat')
    with open(os.path.join(path2CSC_mat, 'channel_numbers_to_names.txt')) as f_channel_names:
        channel_names = f_channel_names.readlines()
        channel_names_dict = dict(zip(map(int, [s.strip().split('\t')[0] for s in channel_names]), [s.strip().split('\t')[1] for s in channel_names]))
    channel_names_dict.pop(0, None) # SKIP THE MIC CHANNEL (channel_num=0)

    # GENERATE A LIST OF SUBLISTS, EACH SUBLIST: [channel_number, channel_name, number_of_cluster_groups[pos], number_of_cluster_groups[neg]]
    channels_with_spikes = []
    if settings.time0 == 0: # BlackRock
        h5_folder = 'CSC_mat' # since combinato clusters based on mat files
    else:
        h5_folder = 'CSC_ncs' # Neuralynx case
    for channel_num, channel_name in channel_names_dict.items():
        h5_files = glob.glob(os.path.join(settings.path2rawdata, 'micro', h5_folder , 'CSC' + str(channel_num), 'data_*.h5'))
        if len(h5_files) == 1: # MAKE SURE THE h5 FILE EXISTS 
            filename = h5_files[0]
            f_all_spikes = h5py.File(filename, 'r')

            num_cluster_groups_pos, num_cluster_groups_neg = (0, 0)
            for sign in signs:
                filename_sorted = glob.glob(os.path.join(settings.path2rawdata, 'micro', 'CSC_ncs', 'CSC' + str(channel_num), 'sort_' + sign + '_yl2', 'sort_cat.h5'))
                if len(filename_sorted) == 1:
                    f_sort_cat = h5py.File(filename_sorted[0], 'r') 
                    groups = f_sort_cat['groups'][()]
                    group_numbers = set([g[1] for g in groups])
                    types = f_sort_cat['types'][()] # -1: artifact, 0: unassigned, 1: MU, 2: SU
                    for g in list(group_numbers):
                        type_of_curr_group = [t_ for (g_, t_) in types if g_ == g]
                        if (len(type_of_curr_group) == 1)|all(t==-1 for t in type_of_curr_group): #sanity check that types has only a single row per group (and exception for artifacts t=-1)
                            type_of_curr_group = type_of_curr_group[0]
                        else:
                            print('file:', filename_sorted[0])
                            #print('Type of curr group:', type_of_curr_group)
                            #print('groups:', groups)
                            #print('types:', types)
                            raise ('issue with types: more than one group assigned to a type')
                        if type_of_curr_group>0:
                            if sign == 'pos':
                                num_cluster_groups_pos += 1
                            if sign == 'neg':
                                num_cluster_groups_neg += 1
                    channels_with_spikes.append([channel_num, channel_name, num_cluster_groups_pos, num_cluster_groups_neg])
                        #if dict_num_group_clusters[sign]>0:
                        #    print('Channel %i (%s) - %s: %i cluster groups' % (channel_num, channel_name, sign, dict_num_group_clusters[sign]))
                     #else:
                     #print('%s was not found!' % os.path.join(settings.path2rawdata, 'micro', 'CSC_ncs', 'CSC' + str(channel_num), 'sort_' + sign + '_yl2', 'sort_cat.h5'))


        else:
            print('None or more than a single combinato h5 was found', channel_num, channel_name, settings.patient, h5_files)

    return channels_with_spikes



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
    sort_IX = np.argsort(events[:, 0], axis=0)
    events = events[sort_IX, :]

    # EVENT_ID dictionary: mapping block names to event numbers
    event_id = dict([(str(event_type_name), event_number[0]) for event_type_name, event_number in zip(event_numbers, event_numbers)])


    return events, event_id





def load_epochs_data(args):
    '''
    Load epochs data from various patients or probe names and return in a list
    
    parameters
    ----------

    args - dict
    dictionary with list of patinets, probe_names, filters, data-types, etc.

    
    Return
    ------

    epochs_list - list
    A list with epochs per patient
    '''
    from utils import load_settings_params
    from utils.utils import probename2picks, pick_responsive_channels
    
    epochs_list = []
    for p, (patient, data_type, filt) in enumerate(zip(args.patient, args.data_type, args.filter)):
        try:
            settings = load_settings_params.Settings(patient)
            fname = '%s_%s_%s_%s-epo.fif' % (patient, data_type, filt, args.level)
            epochs = mne.read_epochs(os.path.join(settings.path2epoch_data, fname), preload=True)
        except: # Exception:
            print(f'WARNING: data not found for {patient} {data_type} {filt} {args.level}')
            continue

        # PICK
        if args.probe_name:
            picks = probename2picks(args.probe_name[p], epochs.ch_names, data_type)
        elif args.channel_name:
            picks = args.channel_name[p]
        elif args.channel_num:
            picks = args.channel_num[p]
        print(picks)
        if not picks:
            print('No picks were found!')
            continue

        epochs.pick(picks)

        # RESPONSIVE CHANNELS
        if args.responsive_channels_only:
            if args.block_type_test: # pick significant channels from both modalities
                block_types = list(set([args.block_type, args.block_type_test]))
            else:
                block_types = [args.block_type] # In next line, list is expected with all block types
            picks = pick_responsive_channels(epochs.ch_names, patient, data_type, filt, block_types , p_value=0.05)
            if picks:
                epochs.pick_channels(picks)
            else:
                print(f'WARNING: No responsive channels were found for {patient} {data_type} {filt} {args.probe_name[p]}')
                continue
      # DECIMATE
        if args.decimate: epochs.decimate(args.decimate)

        metadata = epochs.metadata
        metadata['word_start'] = metadata.apply(lambda row: row['word_string'][0], axis=1)
        metadata['word_end'] = metadata.apply(lambda row: row['word_string'][-1], axis=1)
        epochs.metadata = metadata
        epochs_list.append(epochs)

    return epochs_list


def load_neural_data(args):
    '''
    Generate a epochs list with all neural data based on argparse-user choices 
    '''
    import mne
    from utils import load_settings_params
    from utils.utils import probename2picks, pick_responsive_channels
    from utils.read_logs_and_features import extend_metadata
    # LOAD
    epochs_list = []
    for p, (patient, data_type, filt) in enumerate(zip(args.patient, args.data_type, args.filter)):
        try:
            settings = load_settings_params.Settings(patient)
            fname = '%s_%s_%s_%s-epo.fif' % (patient, data_type, filt, args.level)
            fname = os.path.join(settings.path2epoch_data, fname)
            print(fname)
            epochs = mne.read_epochs(fname, preload=True)
            if 'block_type' in args.__dict__:
                if args.block_type == 'auditory':
                    epochs = epochs['block in [2, 4, 6]'] 
                elif args.block_type == 'visual':
                    epochs = epochs['block in [1, 3, 5]']
            epochs.metadata = extend_metadata(epochs.metadata) # EXTEND METADATA
        except Exception as e: # Data not found
            print(args.__dict__)
            print(fname, str(e))
            # print(f'WARNING: data not found for {patient} {data_type} {filt} {args.level}: {fname}')
            continue
        
        if 'query' in args.__dict__.keys() and args.query:
            print(args.query)
            epochs = epochs[args.query]
        print('Epochs after possible querying:')
        print(epochs)
        # CROP
        if ('tmin' in args.__dict__.keys()) and ('tmax' in args.__dict__.keys()):
            epochs.crop(args.tmin, args.tmax)

        # PICK
        picks = None
        if 'probe_name' in args:
            if args.probe_name:
                probe_name = args.probe_name[p]
                picks = probename2picks(probe_name, epochs.ch_names, data_type)
        if 'channel_name' in args:
            if args.channel_name:
                channel_names = args.channel_name[p]
                picks = channel_names
        if 'channel_num' in args:
            if args.channel_num:
                picks = args.channel_num[p]
        print('picks:', picks)
        # if not picks:
        #     continue

        epochs.pick(picks)

        # RESPONSIVE CHANNELS
        if 'responsive_channels_only' in args:
            if args.responsive_channels_only:
                if args.block_type_test: # pick significant channels from both modalities
                    block_types = list(set([args.block_type, args.block_type_test]))
                else:
                    block_types = [args.block_type] # In next line, list is expected with all block types
                picks = pick_responsive_channels(epochs.ch_names, patient, data_type, filt, block_types , p_value=0.05)
                if picks:
                    epochs.pick_channels(picks)
                else:
                    print(f'WARNING: No responsive channels were found for {patient} {data_type} {filt} {probe_name}')
                    continue


        # DECIMATE
        if 'decimate' in args:
            if args.decimate: epochs.decimate(args.decimate)

        metadata = epochs.metadata
        # metadata['word_start'] = metadata.apply(lambda row: row['word_string'][0], axis=1)
        # metadata['word_end'] = metadata.apply(lambda row: row['word_string'][-1], axis=1)
        epochs.metadata = metadata
        epochs_list.append(epochs)

    return epochs_list


