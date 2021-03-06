import numpy as np
import sys, os, glob, warnings
from utils import load_settings_params


def get_probe_names(patient, micro_macro, path2data='../../../Data/UCLA/'):
    path2channel_names = os.path.join(path2data, 'patient_' + patient, 'Raw', micro_macro, 'CSC_mat', 'channel_numbers_to_names.txt')
    try:
        with open(path2channel_names, 'r') as f:
            channel_names = f.readlines()
        channel_names = [l.strip().split('\t')[1] for l in channel_names]
        if micro_macro == 'micro':
            channel_names.pop(0) # remove MICROPHONE line
            probe_names = [s[4::] for s in channel_names] # remove prefix if exists (in micro: GA1-, GA2-, etc)
        else:
            probe_names = channel_names
        probe_names = [s[:-5] for s in probe_names] # remove file extension and electrode numbering (e.g., LSTG1, LSTG2, LSTG3) 
        if (micro_macro == 'macro') & (patient == 'patient_502'):
            probe_names = [name for name in channel_names if name not in ['ROF', 'RAF']] # 502 has more macro than micro see Notes/log_summary.txt (March 2020)
            print('Macros also include ROF and RAF - see Notes/log_summary.txt (2020Mar02)')
    except:
        print('!!! - Missing %s channel-name files for %s' % (micro_macro, patient))
        return
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
        comp['queries'] = [q+ ' ' + fixed_constraint for q in comp['queries']]

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
    assert data_type in ['micro', 'macro', 'spike'], "Unrecognized data-type (must be 'micro', 'macro' or 'spike')"
    
    if not isinstance(probe_names, list):
        probe_names = [str(probe_names)]
    if 'None' in probe_names: #if None is given to the arg --probe-name then all channels will be picked (i.e., picks=None)
        picks = None
    else:
        picks  = []
        for probe_name in probe_names:
            for ch_name in channel_names:
                if data_type == 'spike':
                    probe_name_from_ch_name = ch_name.split('_')[-1] # assuming formant of the sort p_g6_75_PROBE
                    if probe_name == probe_name_from_ch_name: picks.append(ch_name)
                elif data_type == 'micro':
                    probe_name_from_ch_name = ''.join([i for i in ch_name[4:] if not i.isdigit()]).strip() # remove prefix GA1-, and remove number from ending
                    if probe_name == probe_name_from_ch_name: picks.append(ch_name)
                elif data_type == 'macro':
                    probe_name_from_ch_name = ''.join([i for i in ch_name.split('-')[0] if not i.isdigit()]) # Assuming format of the sort LSTG1-LSTG2 (bipolar)
                    if probe_name == probe_name_from_ch_name: picks.append(ch_name)

    return picks


def pick_responsive_channels(ch_names, patient, data_type, filter_type, block_types, levels=None,  p_value=0.01):
    '''

    Parameters
    ----------
    ch_names: list
    
    patient: str
    
    data_type: str
    One of 'micro', 'macro' or 'spike'
    
    filter_type: str
    One of 'raw', 'gaussian_kernal' or 'high-gamma'
    
    levels: list (default = None)
    One of 'sentence_onset', 'sentence_offset', 'word', 'phone'.
    If None, collect significance from both 'word' and 'sentence_onset'
    
    block_types: list of strings
    With one or more from: 'auditory' or 'visual'
    
    p_value: float
    Threshold for deciding what counts as a responsive channel
    
    Returns
    -------

    channel_names_with_significance: list
    with channel names for which at least one of the clusters has significant p-value
    '''
    
    if levels is None:
        levels = ['word', 'sentence_onset']

    settings = load_settings_params.Settings(patient)
    picks = []
    for level in levels:
        for block_type in block_types:
            fname = f'{patient}_{data_type}_{filter_type}_{level}-epo'
            ext = block_type[0:3] # extention of file is the first 3 letter of block type (vis or aud)
            print(fname+'.'+ext)
            with open(os.path.join(settings.path2epoch_data, fname+'.'+ext), 'r') as f:
                lines = f.readlines()
            lines = lines[4::] # remove header lines
            for line in lines:
                line = line.strip().split(',') # file is comma delimited
                ch_num, ch_name = [l.strip() for l in line[0:2]]
                if line[2]:
                    pval_list = list(map(float, line[2].strip().split(';'))) # p-values are stored as [pval1 pval2 ...]i
                    if any(p<p_value for p in pval_list):
                        picks.append(ch_name)
   
    # take only ch_names with significance
    channel_names_with_significance = [ch_name for ch_name in ch_names if ch_name in picks] 
    
    return channel_names_with_significance


def rescale(data, times, baseline, mode='mean', copy=True, picks=None,

            verbose=None):

    """Rescale (baseline correct) data.



    Parameters

    ----------

    data : array

        It can be of any shape. The only constraint is that the last

        dimension should be time.

    times : 1D array

        Time instants is seconds.

    baseline : tuple or list of length 2, or None

        The time interval to apply rescaling / baseline correction.

        If None do not apply it. If baseline is ``(bmin, bmax)``

        the interval is between ``bmin`` (s) and ``bmax`` (s).

        If ``bmin is None`` the beginning of the data is used

        and if ``bmax is None`` then ``bmax`` is set to the end of the

        interval. If baseline is ``(None, None)`` the entire time

        interval is used. If baseline is None, no correction is applied.

    mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio'

        Perform baseline correction by



        - subtracting the mean of baseline values ('mean')

        - dividing by the mean of baseline values ('ratio')

        - dividing by the mean of baseline values and taking the log

          ('logratio')

        - subtracting the mean of baseline values followed by dividing by

          the mean of baseline values ('percent')

        - subtracting the mean of baseline values and dividing by the

          standard deviation of baseline values ('zscore')

        - dividing by the mean of baseline values, taking the log, and

          dividing by the standard deviation of log baseline values

          ('zlogratio')



    copy : bool

        Whether to return a new instance or modify in place.

    picks : list of int | None

        Data to process along the axis=-2 (None, default, processes all).

    %(verbose)s



    Returns

    -------

    data_scaled: array

        Array of same shape as data after rescaling.

    """

    data = data.copy() if copy else data

    #msg = _log_rescale(baseline, mode)

    #logger.info(msg)

    if baseline is None or data.shape[-1] == 0:

        return data



    bmin, bmax = baseline

    if bmin is None:

        imin = 0

    else:

        imin = np.where(times >= bmin)[0]

        if len(imin) == 0:

            raise ValueError('bmin is too large (%s), it exceeds the largest '

                             'time value' % (bmin,))

        imin = int(imin[0])

    if bmax is None:

        imax = len(times)

    else:

        imax = np.where(times <= bmax)[0]

        if len(imax) == 0:

            raise ValueError('bmax is too small (%s), it is smaller than the '

                             'smallest time value' % (bmax,))

        imax = int(imax[-1]) + 1

    if imin >= imax:

        raise ValueError('Bad rescaling slice (%s:%s) from time values %s, %s'

                         % (imin, imax, bmin, bmax))



    # technically this is inefficient when `picks` is given, but assuming

    # that we generally pick most channels for rescaling, it's not so bad

    mean = np.mean(data[..., imin:imax], axis=-1, keepdims=True)



    if mode == 'mean':

        def fun(d, m):

            d -= m

    elif mode == 'ratio':

        def fun(d, m):

            d /= m

    elif mode == 'logratio':

        def fun(d, m):

            d /= m

            np.log10(d, out=d)

    elif mode == 'percent':

        def fun(d, m):

            d -= m

            d /= m

    elif mode == 'zscore':

        def fun(d, m):

            d -= m

            d /= np.std(d[..., imin:imax], axis=-1, keepdims=True)

    elif mode == 'zlogratio':

        def fun(d, m):

            d /= m

            np.log10(d, out=d)

            d /= np.std(d[..., imin:imax], axis=-1, keepdims=True)



    if picks is None:

        fun(data, mean)

    else:

        for pi in picks:

            fun(data[..., pi, :], mean[..., pi, :])

    return data
