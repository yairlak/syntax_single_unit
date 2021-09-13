import os
import numpy as np
import warnings
from utils import load_settings_params


def get_probe_names(patient, data_type, path2data='../../../Data/UCLA/'):
    path2channel_names = os.path.join(path2data, 'patient_' + patient, 'Raw', data_type, 'channel_numbers_to_names.txt')
    #print(path2channel_names)
    if data_type =='microphone':
        return ['MICROPHONE'], ['MICROPHONE']
    try:
        with open(path2channel_names, 'r') as f:
            channel_names = f.readlines()
        #print(channel_names)
        channel_names = [l.strip().split()[1] for l in channel_names]
        if data_type == 'micro':
            probe_names = [s[4::] if s.startswith('G') else s for s in channel_names] # remove prefix if exists (in micro: GA1-, GA2-, etc)
            probe_names = [s[:-1] for s in probe_names] # remove file extension and electrode numbering (e.g., LSTG1, LSTG2, LSTG3) 
        elif data_type == 'macro':
            probe_names = []
            for ch_name in channel_names:
                IX_dash = ch_name.index('-')
                probe_name = ch_name[:(IX_dash-1)] # remove dash *and* channel numbering
                probe_names.append(probe_name)
        elif data_type == 'spike':
            #  e.g., GB1-RASTG8_40p1
            probe_names = [s[4::].split('_')[0][:-1] for s in channel_names]

        if (data_type == 'macro') & (patient == 'patient_502'):
            probe_names = [name for name in channel_names if name not in ['ROF', 'RAF']] # 502 has more macro than micro see Notes/log_summary.txt (March 2020)
            print('Macros also include ROF and RAF - see Notes/log_summary.txt (2020Mar02)')
    except:
        print('!!! - Missing %s channel-name files for %s' % (data_type, patient))
        return [], []
    print(list(set(probe_names)))
    return sorted(list(set(probe_names))), channel_names


def get_queries(comparison):
    str_blocks = ['block == {} or '.format(block) for block in eval(comparison['blocks'])]
    str_blocks = '(' + ''.join(str_blocks)[0:-4] + ')'

    if comparison['align_to'] == 'FIRST':
        str_align = 'word_position == 1'
    elif comparison['align_to'] == 'LAST':
        str_align = 'word_position == sentence_length'
    elif comparison['align_to'] == 'END':
        str_align = 'word_position == -1'
    elif comparison['align_to'] == 'EACH':
        str_align = 'word_position > 0'

    queries = []
    for query_cond, label_cond in zip(comparison['query'], comparison['cond_labels']):
        # If pos in query then add double quotes (") around value, e.g. (pos==VB --> pos =="VB")
        new_query_cond = ''
        i = 0
        while i < len(query_cond):
            if query_cond[i:i + len('pos==')] == 'pos==':
                reminder = query_cond[i + len('pos==')::]
                temp_list = reminder.split(" ", 1)
                new_query_cond = new_query_cond + 'pos=="' + temp_list[0] + '" '
                i = i + 6 + len(temp_list[0])
            else:
                new_query_cond += query_cond[i]
                i += 1
            query_cond = new_query_cond

        queries.append(query_cond + ' and ' + str_align + ' and ' + str_blocks)

    return queries


def update_queries(comp, block_type, fixed_constraint, metadata):
    # If 'queries' value is a string (instead of a list of strings) then queries are based on all values in metadata
    # The string in 'queries' should indicate a field name in metadata.
    if isinstance(comp['queries'], str):
        if comp['queries'] != 'phone_string':
            queries = []; condition_names = []
            all_possible_values = sorted(list(set(metadata[comp['queries']])))
            
            for val in all_possible_values:
                if isinstance(val, str):
                    query = comp['queries'] + ' == "' + str(val) + '"'
                else:
                    query = comp['queries'] + ' == ' + str(val)
                queries.append(query)
                condition_names.append(str(val))
            comp['queries'] = queries
            comp['condition_names'] = condition_names
        else: # special case for phone_string
            queries = []; condition_names = []
            all_possible_values = list(set(metadata[comp['queries']]))
            print(all_possible_values)
            for val in all_possible_values:
                if isinstance(val, str):
                    if all([not i.isdigit() for i in val[0:-1]]) and val[-1].isdigit(): # if only last character is a digit
                        val = val[0:-1]
                        query = comp['queries'] + '.str.startswith("' + str(val) + '")'
                    else:
                        query = comp['queries'] + ' == "' + str(val) + '"'
                    if query not in queries:
                        queries.append(query)
                        condition_names.append(str(val))
            assert len(queries) == len(condition_names)
            comp['queries'] = queries
            comp['condition_names'] = condition_names

    if not comp['colors']:
        color_vals = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for i, _ in enumerate(comp['queries']):
            comp['colors'].append(color_vals[ i % 7])

    # Add block constraint
    comp['block_type'] = block_type
    if block_type:
        block_str = ' and (block in [1, 3, 5])' if block_type == 'visual' else ' and (block in [2, 4, 6])'
        comp['queries'] = [q+block_str for q in comp['queries']]

    # Add fixed constraint if provided in args (for example, to limit to first phone in auditory blocks)
    if fixed_constraint:
        comp['fixed_constraint'] = fixed_constraint
        comp['queries'] = [f'({q}) and ({fixed_constraint})' for q in comp['queries']]

    return comp


def dict2filename(d, sep='_', keys_to_use=[], extension='', show_values_only=False):
    '''
    This function generates a filename that contains chosen keys-values pairs from a dictionary.
    For example, the dict can represent hyperparameters of a model or settings.
    USAGE EXAMPLE:
    filename = get_filename_from_dict(my_dict, '-', ['frequency_band', 'smoothing'])
    :param d: (dict) keys and corresponding values to be used in generating the filename.
    :param sep: (str) separator to use in filename, between keys and values of d.
    :param keys_to_use: (list) subset of keys of d. If empty then all keys in d will be appear in filename.
    :param extension: (str) extension of the file name (e.g., 'csv', 'png').
    :return: (str) the filename generated from key-value pairs of d.
    '''

    # Check if dictionary is empry
    if not d:
        raise('Dictionary is empty')
    # filter the dictionary based on keys_to_use
    if keys_to_use:
        for i, k in enumerate(keys_to_use):
            if k not in d.keys():
                warnings.warn('Chosen key (%s) is not in dict' % k)
                keys_to_use.pop(i)
    else:
        keys_to_use = sorted(d.keys())

    # add extension
    if len(extension) > 0:
        extension = '.' + extension
    
    if show_values_only:
        l = []
        for k in keys_to_use:
            if isinstance(d[k], list) and d[k]:
                if isinstance(d[k][0], list): # nested list
                    curr_str = sep.join([str(item) for sublist in d[k] for item in sublist])
                    l.append(curr_str)
                else: # list
                    l.append(sep.join([str(s) for s in d[k]]))
            else: # not a list
                l.append(str(d[k])) 
        fn = sep.join(l) + extension
    else:
        fn = sep.join([sep.join((str(k), str(d[k]))) for k in keys_to_use]) + extension
    return fn


def probename2picks(probe_names, channel_names, data_type):
    '''
    parameters
    ----------
    probe_names : list
    containing strings of probe_names to be picked

    channel_names: list
    the entire list of channel-string names from which probe should be picked

    data_type: str
    either 'micro', 'macro', or 'spike'
    '''
    assert data_type in ['micro', 'macro', 'spike', 'microphone'], "Unrecognized data-type (must be 'micro', 'macro' or 'spike')"
    #print(probe_names, channel_names)
    
    if not isinstance(probe_names, list):
        probe_names = [str(probe_names)]
    if 'None' in probe_names: #if None is given to the arg --probe-name then all channels will be picked (i.e., picks=None)
        picks = None
    else:
        picks  = []
        for probe_name in probe_names:
            for ch_name in channel_names:
                if data_type == 'spike':
                    if '-' in ch_name: # assuming formant of the sort {ch_name}_{ch_num}{sign}{group_num} (e.g., GA1-LAH1_2p1
                        probe_name_from_ch_name = ch_name.split('-')[1].split('_')[0]
                    else: # assuming formant of the sort p_g6_75_PROBE
                        probe_name_from_ch_name = ch_name.split('_')[-1] 
                    # remove digits
                    probe_name_from_ch_name = ''.join([i for i in probe_name_from_ch_name if not i.isdigit()])
                    if probe_name == probe_name_from_ch_name: picks.append(ch_name)
                elif data_type == 'micro':
                    probe_name_from_ch_name = ''.join([i for i in ch_name[4:] if not i.isdigit()]).strip() # remove prefix GA1-, and remove number from ending
                    #print(probe_name_from_ch_name)
                    if probe_name == probe_name_from_ch_name: picks.append(ch_name)
                elif data_type == 'macro':
                    probe_name_from_ch_name = ''.join([i for i in ch_name.split('-')[0] if not i.isdigit()]) # Assuming format of the sort LSTG1-LSTG2 (bipolar)
                    if probe_name == probe_name_from_ch_name: picks.append(ch_name)

    return picks


