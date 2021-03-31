import os, glob
from scipy import io
import numpy as np
import re

# Raw data
#def micro_electrodes_raw(settings):
#    if settings.channel > 0:
#        CSC_file = glob.glob(os.path.join(settings.path2rawdata_mat, 'CSC' + str(settings.channel) + '.mat'))
#    elif settings.channel == 0:
#        CSC_file = glob.glob(os.path.join(settings.path2rawdata_mat, 'MICROPHONE.mat'))
#    print(CSC_file)
#    data_all = io.loadmat(CSC_file[0])['data']
#    print('channel-data loaded')
#    if 'file_name' in io.loadmat(CSC_file[0]).keys():
#        settings.channel_name = io.loadmat(CSC_file[0])['file_name'][0]
#    else:
#        settings.channel_name = 'Channel_'+str(settings.channel)
#    return data_all, settings


#def macro_electrodes(settings):
#    CSC_file = glob.glob(os.path.join(settings.path2macro, 'CSC' + str(settings.channel_macro) + '.mat'))
#    data_all = io.loadmat(CSC_file[0])['data']
#    if 'file_name' in io.loadmat(CSC_file[0]).keys():
#        settings.channel_name = io.loadmat(CSC_file[0])['file_name'][0]
#    else:
#        settings.channel_name = 'Channel_' + str(settings.channel_macro)
#    return data_all, settings



def load_combinato_sorted_h5(channel_num, channel_name, settings):
    import h5py
    spike_times = []; channel_names = []

    h5_files = glob.glob(os.path.join(settings.path2rawdata, 'micro', 'CSC_ncs' , 'CSC' + str(channel_num), 'data_*.h5'))
    if len(h5_files) == 1:
        filename = h5_files[0]
        f_all_spikes = h5py.File(filename, 'r')

        for sign in ['pos', 'neg']:
            filename_sorted = glob.glob(os.path.join(settings.path2rawdata, 'micro', 'CSC_ncs', 'CSC' + str(channel_num), 'sort_' + sign + '_simple', 'sort_cat.h5'))[0]
            f_sort_cat = h5py.File(filename_sorted, 'r')

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
                    spike_times.append(curr_spike_times)
                    region_name = channel_name[1+channel_name.find("-"):channel_name.find(".")]
                    channel_names.append(sign[0] + '_g' + str(g) + '_' + str(channel_num)+ '_' + region_name)
        print(channel_num)


    else:
        print('None or more than a single combinato h5 was found')

    return spike_times, channel_names


# Spike-sorted data
def spike_clusters(settings):
    CSC_cluster_files = glob.glob(os.path.join(settings.path2spike_clusters, 'CSC*_cluster.mat'))
    CSC_cluster_files.sort(key = alphanum_key)
    data_all = []; electrode_names_from_raw_files = []; from_channels = []
    for cluster in CSC_cluster_files:
        data_all.append(io.loadmat(cluster)['spike_times_sec'])
        settings.time0 = io.loadmat(cluster)['time0'][0,0]
        settings.timeend = io.loadmat(cluster)['timeend'][0,0]
        #print(io.loadmat(cluster).keys())
        if 'from_channel' in io.loadmat(cluster).keys():
            electrode_names_from_raw_files.append(io.loadmat(cluster)['electrode_name'][0] + ',ch ' + str(io.loadmat(cluster)['from_channel'][0]))
            from_channels.append(io.loadmat(cluster)['from_channel'][0][0])

    if not electrode_names_from_raw_files:
        electrode_names_from_raw_files = cluster_to_electrode_name(settings)

    return data_all, settings, electrode_names_from_raw_files, from_channels


def wave_clus_output_files(settings):
    # Find all times_*.mat files in the output folder from Wave_clus
    times_files = glob.glob(os.path.join(settings.path2rawdata_mat, 'times_CSC*.mat'))

    # Extract the numbers of the channels found
    channel_numbers = []
    data_all_channels_spike_clusters = [None] * 1000 # Assuming never more than 1000 channels
    for channel_filename in times_files:
        curr_channel_number = int(''.join([s for s in os.path.basename(channel_filename) if s.isdigit()]))
        channel_numbers.append(curr_channel_number)
        # ZERO-based indexing
        data_all_channels_spike_clusters[curr_channel_number-1] = io.loadmat(channel_filename)['cluster_class']
        try:
            settings.time0 = io.loadmat(channel_filename)['time0'][0,0]
            settings.timeend = io.loadmat(channel_filename)['timeend'][0,0]
        except:
            print(channel_filename)

    return data_all_channels_spike_clusters, channel_numbers, settings

# Auxilary functions
def cluster_to_electrode_name(settings):
    with open(os.path.join(settings.path2log, 'clusters_electrode_montage.m')) as f:
        electrode_names = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        electrode_names = [x.strip().split("\t") for x in electrode_names if "\t" in x]
    for ele in range(len(electrode_names)):
        electrode_names[ele][0] = electrode_names[ele][0][:-1]
        electrode_names[ele][1] = ''.join([x for x in electrode_names[ele][1] if (x != "'" and x != ",")])
    electrode_names_list = [None] * 1000
    for ele in range(len(electrode_names)):
        for IX in electrode_names[ele][0].split(":"):
            electrode_names_list[int(IX)-1] = electrode_names[ele][1]

    electrode_names_list = [s for s in electrode_names_list if s]

    return electrode_names_list


def electrodes_names(settings):
    electrode_names = io.loadmat(os.path.join(settings.path2patient_folder, 'electrodes_info_names.mat'))['electrodes_info'][0]
    electrode_names = [s[0] for s in electrode_names]
    return electrode_names


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]
