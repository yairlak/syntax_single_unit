import argparse
import os
import pickle
import mne
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from plotting.viz_ERPs import get_sorting_IXs, average_repeated_trials
from utils import comparisons, load_settings_params
from utils.data_manip import DataHandler
from utils.utils import probename2picks, update_queries
from scipy.ndimage import gaussian_filter1d
from spykes.plot.neurovis import NeuroVis
from spykes.ml.neuropop import NeuroPop

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(description='Generate trial-wise plots')
# DATA
parser.add_argument('--patient', default='479_11', help='Patient string')
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike', 'microphone'],
                    default='spike', help='electrode type')
parser.add_argument('--level', choices=['sentence_onset', 'sentence_offset',
                                        'word', 'phone'],
                    default='word', help='')
parser.add_argument('--filter', default='raw', help='')
parser.add_argument('--smooth', default=None, help='')
parser.add_argument('--scale-epochs', action="store_true", default=False, help='')
# PICK CHANNELS
parser.add_argument('--probe-name', default=None, nargs='*', type=str,
                    help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=None, nargs='*', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
# QUERY (SELECT TRIALS)
parser.add_argument('--comparison-name', default= '479_11_LSTG7_15p2', help='int. Comparison name from Code/Main/functions/comparisons_level.py. see print_comparisons.py')
parser.add_argument('--block-type', default=[], help='Block type will be added to the query in the comparison')
parser.add_argument('--fixed-constraint', default=[], help='A fixed constrained added to query. For example first_phone == 1 for auditory blocks')
parser.add_argument('--average-repeated-trials', action="store_true", default=False, help='')
parser.add_argument('--tmin', default=-1, type=float, help='crop window. If empty, only crops 0.1s from both sides, due to edge effects.')
parser.add_argument('--tmax', default=0.6, type=float, help='crop window')
parser.add_argument('--baseline', default=[], type=str, help='Baseline to apply as in mne: (a, b), (None, b), (a, None), (None, None) or None')
parser.add_argument('--baseline-mode', choices=['mean', 'ratio', 'logratio', 'percent', 'zscore', 'zlogratio'], default=None, help='Type of baseline method')
# MISC
parser.add_argument('--SOA', default=500, help='SOA in design [msec]')
parser.add_argument('--word-ON-duration', default=250, help='Duration for which word word presented in the RSVP [msec]')
parser.add_argument('--remove-outliers', action="store_true", default=False, help='Remove outliers based on percentile 25 and 75')
parser.add_argument('--no-title', action="store_true", default=False)
parser.add_argument('--yticklabels-sortkey', type=int, default=[], help="")
parser.add_argument('--xticklabels-fontsize', type=int, default=20, help="")
parser.add_argument('--yticklabels-fontsize', type=int, default=20, help="")
parser.add_argument('--dont-write', default=False, action='store_true', help="If True then file will be overwritten")
parser.add_argument('--sort-key', default=['word_string'], help='Keys to sort according')
parser.add_argument('--y-tick-step', default=10, type=int, help='If sorted by key, set the yticklabels density')
parser.add_argument('--window-st', default=50, type=int, help='Regression start-time window [msec]')
parser.add_argument('--window-ed', default=450, type=int, help='Regression end-time window [msec]')
parser.add_argument('--vmin', default=-2.5, help='vmin of plot (default is in zscore, assuming baseline is zscore)')
parser.add_argument('--vmax', default=2.5, help='vmax of plot (default is in zscore, assuming baseline is zscore')
parser.add_argument('--smooth-raster', default=0.002, help='If empty no smoothing. Else, gaussian width in [sec], assuming sfreq=1000Hz')
parser.add_argument('--save2', default=[], help='If empty saves figure to default folder')


args = parser.parse_args()

assert not (args.data_type == 'spike' and args.scale_epochs == True)
args.patient = 'patient_' + args.patient
if isinstance(args.sort_key, str):
    args.sort_key = eval(args.sort_key)
if isinstance(args.baseline, str):
    args.baseline = eval(args.baseline)
if args.data_type == 'spike':
    args.vmin = 0
    args.vmax = 0.5
print(args)


comparisons = comparisons.comparison_list()
comparison = comparisons[args.comparison_name].copy()
if 'channel_name' in comparison.keys():
    args.channel_name = [comparison['channel_name']]

# LOAD
data = DataHandler(args.patient, args.data_type, args.filter,
                   args.probe_name, args.channel_name, args.channel_num)
# Both neural and feature data into a single raw object
data.load_raw_data(verbose=True)

# COMPARISON

if 'level' in comparison.keys():
    args.level = comparison['level']

# GET SENTENCE-LEVEL DATA BEFORE SPLIT
data.epoch_data(level=args.level,
                query=None,
                scale_epochs=args.scale_epochs,
                smooth=args.smooth,
                verbose=True)
epochs = data.epochs[0]
metadata = epochs.metadata.copy()
metadata['sentence_string'] = metadata.apply(lambda row: row['sentence_string'].capitalize(), axis=1)


if 'sort' not in comparison.keys():
    comparison['sort'] = args.sort_key
else:
    print(comparison['sort'])

if comparison['sort']:
    str_sort = 'Trials are sorted by:%s' % comparison['sort'][0]
else:
    str_sort = ''
            
if 'tmin_tmax' in comparison.keys():
    args.tmin, args.tmax = comparison['tmin_tmax']
if 'y-tick-step' in comparison.keys():
    args.y_tick_step = comparison['y-tick-step']
if 'fixed_constraint' in comparison.keys():
    args.fixed_constraint = comparison['fixed_constraint']


tmin, tmax = args.tmin*1e3, args.tmax*1e3

comparison = update_queries(comparison, args.block_type,
                            args.fixed_constraint, epochs.metadata)

print(args.comparison_name)
pprint(comparison)
dict_prop = {}
for i_cond, cond_name in enumerate(comparison['condition_names']):
    dict_prop[cond_name] = {}
    for prop in ['colors', 'cmaps', 'ls', 'y-tick-step']:
        if prop in comparison.keys():
            dict_prop[cond_name][prop] = comparison[prop][i_cond]

# n_conds = len(comparison['queries'])

# PICK
if args.probe_name:
    picks = probename2picks(args.probe_name, epochs.ch_names, args.data_type)
    epochs.pick_channels(picks)
elif args.channel_name:
    epochs.pick_channels(args.channel_name)
elif args.channel_num:
    epochs.pick(args.channel_num)


print('-'*100)
print(epochs.ch_names)

stimulus_strings = []
# metadata['comparison'] = 'Other'
queries = []
for condition_name, query in zip(comparison['condition_names'], comparison['queries']):
    IXs = metadata.query(query, engine='python').index
    metadata.loc[IXs, 'comparison'] = condition_name
    stimulus_strings.append(metadata.loc[IXs, 'word_string'])
    queries.append(query)
metadata = metadata.query("("+ ")|(".join(queries) + ")", engine='python')




for ch, ch_name in enumerate(epochs.ch_names):
    print(ch_name)
    
    
    # SPIKING DATA    
    IXs = data.raws[0].get_data()[ch, :].astype('bool')
    spike_times = data.raws[0].times[IXs]
    curr_neuron = NeuroVis(spike_times, name=f'{args.patient}, {ch_name}\n{args.comparison_name}\n {str_sort}')
    
    
    # get rasters
    rasters = curr_neuron.get_raster(event='event_time',
                                    conditions='comparison',
                                    df=metadata,
                                    window=[tmin, tmax],
                                    binsize=5,
                                    plot=False)
    
    # plot rasters
    
    
    
    if 'figsize' in comparison.keys():
        figsize = comparison['figsize']
    else:
        figsize = (4, 20)
    
    
    gridpsec_kw = {}
    if 'height_ratios' in comparison.keys():
        height_ratios = []
        for cond_id in comparison['condition_names']:
            if comparison['height_ratios']:
                n_trials_curr_cond = rasters['data'][cond_id].shape[0]
            else:
                n_trials_curr_cond = 1
            height_ratios.append(n_trials_curr_cond)
        gridpsec_kw['height_ratios'] = height_ratios
        if comparison['height_ratios']:
                height_ratios.append(int(sum(height_ratios)/3))
        else:
            height_ratios.append(1)
    
    
    n_conds = len(list(rasters['data'].keys()))
    fig, axs = plt.subplots(n_conds+1, 1, figsize=figsize, gridspec_kw=gridpsec_kw)
    for i, cond_id in enumerate(sorted(list(rasters['data'].keys()))):
        IX_subplot = comparison['condition_names'].index(cond_id)
        #plt.subplot(n_conds+1,1, IX_subplot+1)
        plt.sca(axs[IX_subplot])
        if IX_subplot==0:
            has_title=False # True
        else:
            has_title=False

        # SORTING
        if comparison['sort'] not in ['rate', 'latency']:
            sortby, yticklabels, fields_for_sorting = get_sorting_IXs(metadata,
                                                                        cond_id,
                                                                        comparison['sort'],
                                                                        ch_name,
                                                                        args,
                                                                        join_ticklabels=True)
        else:
            sortby = comparison['sort']
        

        if 'cmaps' in comparison.keys():
            curr_cmap = dict_prop[cond_id]['cmaps']
        else:
            curr_cmap = None
        if 'colors' in comparison.keys():
            curr_color = dict_prop[cond_id]['colors']
        else:
            curr_color = None
        
        IXs_sort = curr_neuron.plot_raster(rasters,
                                           cond_id=cond_id,
                                           cmap=curr_cmap,
                                           cond_name=f'cond_id',
                                           sortby=sortby,
                                           sortorder='ascend',
                                           has_title=has_title)
        
        if comparison['sort'] in ['rate', 'latency']:
            if args.level == 'word':
                field = 'word_string'
            elif args.level in ['sentence_onset', 'sentence_offset']:
                field = 'sentence_string'
            yticklabels = metadata[metadata['comparison'] == cond_id][field].to_numpy()
            yticklabels = yticklabels[IXs_sort]
        
        # generate corresponding ticks
        yticks = range(len(yticklabels))
        # Subsample tick labels
        if 'y-tick-step' in dict_prop[list(dict_prop.keys())[0]].keys():
            yticks = yticks[::dict_prop[cond_id]['y-tick-step']]
            yticklabels = yticklabels[::dict_prop[cond_id]['y-tick-step']]
        # Replace ticks
        plt.gca().set_yticks(yticks)
        plt.gca().tick_params(axis='x', which='major', labelsize=20)
        plt.gca().set_yticklabels(yticklabels, fontsize=args.yticklabels_fontsize)
        #plt.gca().set_ylabel(cond_id, fontsize=20)
        if curr_color is not None:
            #plt.gca().yaxis.label.set_color(curr_color)
            plt.gca().tick_params(axis='y', colors=curr_color)
        
        
        # if i < n_conds-1:
        plt.xlabel('')
    
    #plt.subplot(n_conds+1,1, n_conds+1)
    plt.sca(axs[-1])
    if 'ylim' in comparison.keys():
        ylim = comparison['ylim']   
    else:
        ylim = 70
    
    sorted_colors = np.asarray(comparison['colors'])[np.argsort(comparison['condition_names'])].tolist()
    psth_M1 = curr_neuron.get_psth(event='event_time',
                                 df=metadata,
                                 conditions='comparison',
                                 window=[tmin, tmax],
                                 binsize=5,
                                 plot=True,
                                 ylim=[0, ylim],
                                 colors=sorted_colors)
                                 #event_name='Block Type')
    plt.title('')
    plt.xlabel('')
    plt.ylabel('Firing Rate [Hz]', fontsize=20)
    plt.gca().tick_params(axis='both', which='major', labelsize=20)
    plt.subplots_adjust(hspace=-1)
    #plt.tight_layout()
   
    ##############
    # ERP FIGURE #
    ##############
    
    label_y = 'firing rate (Hz)'
    ylim = [-1, 20]
    # ylim = [None, None]
    yticks = [0, 10, 20, 30, 40]
    
    ########
    # SAVE #
    ########
    plt.tight_layout()
    if isinstance(comparison['sort'], list):
        comparison_str = '_'.join(comparison['sort'])
    else:
        comparison_str = comparison['sort']
    
    
    if not args.save2:
        if isinstance(comparison['sort'], list):
            comparison_str = '_'.join(comparison['sort'])
        else:
            comparison_str = comparison['sort']
        fname_fig = 'ERP_trialwise_%s_%s_%s_%s_%s_%s_%s_%s_%s' % (args.patient, args.data_type, args.level, args.filter, args.smooth, ch_name, args.block_type, args.comparison_name, comparison_str)
        if args.average_repeated_trials:
            fname_fig += '_lumped'
        if args.fixed_constraint:
            fname_fig += '_'+args.fixed_constraint
        fname_fig += '.png'
        if args.responsive_channels_only:
            dname_fig = os.path.join('..', '..', 'Figures', 'Comparisons', 'responsive', args.comparison_name, args.patient, 'ERPs', args.data_type)
        else:
            dname_fig = os.path.join('..', '..', 'Figures', 'Comparisons', args.comparison_name, args.patient, 'ERPs', args.data_type)
        if not os.path.exists(dname_fig):
            os.makedirs(dname_fig)
        fname_fig = os.path.join(dname_fig, fname_fig)
    else:
        fname_fig = args.save2

    
    fig.savefig(fname_fig)
    print('fig saved to: %s' % fname_fig)
    plt.close()
