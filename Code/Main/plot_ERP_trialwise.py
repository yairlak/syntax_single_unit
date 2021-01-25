import argparse, os, glob, sys
# Set current working directory to that of script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
#sys.path.append('..')
import mne
import matplotlib.pyplot as plt
import numpy as np
from numpy import percentile
from sklearn import linear_model
from sklearn.metrics import r2_score
from operator import itemgetter
from pprint import pprint
from functions.read_logs_and_features import extend_metadata
from functions import comparisons, load_settings_params
from functions.utils import probename2picks, rescale, update_queries, pick_responsive_channels
from scipy.ndimage import gaussian_filter1d

parser = argparse.ArgumentParser(description='Generate plots for TIMIT experiment')
parser.add_argument('--patient', default='479_11', help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], default='micro', help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='sentence_onset', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'gaussian-kernel-25', 'high-gamma'], default='high-gamma', help='')
parser.add_argument('--probe-name', default=[], nargs='*', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=[], nargs='*', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
parser.add_argument('--comparison-name', default=[], help='int. Comparison name from Code/Main/functions/comparisons_level.py. see print_comparisons.py')
parser.add_argument('--block-type', default=[], help='Block type will be added to the query in the comparison')
parser.add_argument('--fixed-constraint', default=[], help='A fixed constrained added to query. For example first_phone == 1 for auditory blocks')
parser.add_argument('--tmin', default=[], type=float, help='crop window. If empty, only crops 0.1s from both sides, due to edge effects.')
parser.add_argument('--tmax', default=[], type=float, help='crop window')
parser.add_argument('--baseline', default=[], type=str, help='Baseline to apply as in mne: (a, b), (None, b), (a, None), (None, None) or None')
parser.add_argument('--baseline-mode', choices=['mean', 'ratio', 'logratio', 'percent', 'zscore', 'zlogratio'], default='zscore', help='Type of baseline method')
parser.add_argument('--SOA', default=500, help='SOA in design [msec]')
parser.add_argument('--word-ON-duration', default=250, help='Duration for which word word presented in the RSVP [msec]')
parser.add_argument('--remove-outliers', action="store_true", default=False, help='Remove outliers based on percentile 25 and 75')
parser.add_argument('--no-title', action="store_true", default=False)
parser.add_argument('--yticklabels-sortkey', type=int, default=[], help="")
parser.add_argument('--yticklabels-fontsize', type=int, default=14, help="")
parser.add_argument('--dont-write', default=False, action='store_true', help="If True then file will be overwritten")
parser.add_argument('--sort-key', default=['sentence_length'], help='Keys to sort according')
parser.add_argument('--y-tick-step', default=30, type=int, help='If sorted by key, set the yticklabels density')
parser.add_argument('--window-st', default=50, type=int, help='Regression start-time window [msec]')
parser.add_argument('--window-ed', default=450, type=int, help='Regression end-time window [msec]')
parser.add_argument('--vmin', default=-2.5, help='vmin of plot (default is in zscore, assuming baseline is zscore)')
parser.add_argument('--vmax', default=2.5, help='vmax of plot (default is in zscore, assuming baseline is zscore')
parser.add_argument('--smooth-raster', default=0.002, help='If empty no smoothing. Else, gaussian width in [sec], assuming sfreq=1000Hz')
parser.add_argument('--save2', default=[], help='If empty saves figure to default folder')


args = parser.parse_args()
args.patient = 'patient_' + args.patient
if isinstance(args.sort_key, str):
    args.sort_key = eval(args.sort_key)
if isinstance(args.baseline, str):
    args.baseline = eval(args.baseline)
if args.data_type == 'spike':
    args.vmin = 0
    args.vmax = 1
print(args)

print('Loading settings...')
settings = load_settings_params.Settings(args.patient)

# LOAD
fname = '%s_%s_%s_%s-epo.fif' % (args.patient, args.data_type, args.filter, args.level)
epochs = mne.read_epochs(os.path.join(settings.path2epoch_data, fname), preload=True)
#epochs.metadata = extend_metadata(epochs.metadata)

# COMPARISON
comparisons = comparisons.comparison_list()
comparison = comparisons[args.comparison_name].copy()
comparison = update_queries(comparison, args.block_type, args.fixed_constraint, epochs.metadata)
pprint(comparison)

print(args.comparison_name)
for query in comparison['queries']: print(query)

if 'sort' not in comparison.keys():
    comparison['sort'] = args.sort_key
else:
    print(comparison['sort'])

if 'tmin_tmax' in comparison.keys():
    args.tmin, args.tmax = comparison['tmin_tmax']
if 'y-tick-step' in comparison.keys():
    args.y_tick_step = comparison['y-tick-step']

# PICK
if args.probe_name:
    picks = probename2picks(args.probe_name, epochs.ch_names, args.data_type)
    epochs.pick_channels(picks)
elif args.channel_name:
    epochs.pick_channels(args.channel_name)
elif args.channel_num:
    epochs.pick(args.channel_num)

# Filter non-responsive channels
if args.responsive_channels_only:
    if args.data_type == 'spike':
        filt = 'gaussian-kernel' # if spikes, plot raw but calc signficance for responsivness based on smoothed data.
    else:
        filt = args.filter
    picks = pick_responsive_channels(epochs.ch_names, args.patient, args.data_type, filt, [args.block_type], p_value=0.01)
    if picks:
        epochs.pick_channels(picks)
    else:
        raise('No responsive channels were found')


print('-'*100)
print(epochs.ch_names)

# BASELINE
if args.filter != 'high-gamma': # high-gamma is already baselined during epoching (generate_epochs.py)
    if args.baseline:
        print('Apply baseline:')
        epochs.apply_baseline(args.baseline, verbose=True)
else: #baseline high-gamma (e.g., to dB)
    pass
    #if args.baseline and args.baseline_mode:
    #    epochs._data = rescale(epochs.get_data(), epochs.times, args.baseline, args.baseline_mode) 
#print(epochs._data[:2, :100])

# CROP
if args.tmin and args.tmax:
    epochs.crop(args.tmin, args.tmax)
else:
    if args.filter == 'high-gamma': # remove boundary effects
        epochs.crop(min(epochs.times) + 0.1, max(epochs.times) - 0.1)

for ch, ch_name in enumerate(epochs.ch_names):
    print(ch_name)
    if ch_name == 'MICROPHONE': continue
    # output filename of figure
    str_comparison = '_'.join([tup[0] for tup in comparison['queries']])
    if not args.save2:
        fname_fig = 'ERP_trialwise_%s_%s_%s_%s_%s_%s_%s_%s.png' % (args.patient, args.data_type, args.level, args.filter, ch_name, args.block_type, args.comparison_name, '_'.join(comparison['sort']))
        if args.responsive_channels_only:
            dname_fig = os.path.join(settings.path2figures, 'Comparisons', 'responsive', args.comparison_name, args.patient, 'ERPs', args.data_type)
        else:
            dname_fig = os.path.join(settings.path2figures, 'Comparisons', args.comparison_name, args.patient, 'ERPs', args.data_type)
        if not os.path.exists(dname_fig):
            os.makedirs(dname_fig)
        fname_fig = os.path.join(dname_fig, fname_fig)
    else:
        fname_fig = args.save2

    if (not os.path.exists(fname_fig)) or (not args.dont_write): # Check if output fig file already exists: 
        # Get number of trials from each query
        nums_trials = []; ims = []
        for query in comparison['queries']:
            data_curr_query = epochs[query].pick(ch_name).get_data()[:, 0, :]
            nums_trials.append(data_curr_query.shape[0]) # query and pick channel
        print('Number of trials from each query:', nums_trials)
        nums_trials_cumsum = np.cumsum(nums_trials)
        nums_trials_cumsum = [0] + nums_trials_cumsum.tolist()
        # Prepare subplots
        fig, _ = plt.subplots(figsize=(15, 10))
        num_queries = len(comparison['queries'])
        height_ERP = int(np.ceil(sum(nums_trials)/num_queries))
        if num_queries > 1:
            spacing = int(np.ceil(0.1*sum(nums_trials)/num_queries))
        else:
            spacing = 0
        nrows = sum(nums_trials)+height_ERP+spacing*num_queries; ncols = 10 # number of rows in subplot grid per query. Width is set to 10. num_queries is added for 1-row spacing
        # prepare axis for ERPs 
        ax2 = plt.subplot2grid((nrows, ncols+1), (sum(nums_trials)+spacing*num_queries, 0), rowspan=height_ERP, colspan=10) # Bottom figure for ERP
        # Collect data from all queries and sort based on args.sort_key
        data = []
        evoked_dict = dict()
        colors_dict = {}
        linestyles_dict = {}
        for i_query, query in enumerate(comparison['queries']):
            condition_name = comparison['condition_names'][i_query]
            height_query_data = nums_trials[i_query]
            color = comparison['colors'][i_query]
            colors_dict[condition_name] = color
            if 'ls' in comparison.keys():
                ls = comparison['ls'][i_query]
                linestyles_dict[condition_name] = ls
            data_curr_query = epochs[query].pick(ch_name).get_data()[:, 0, :] # query and pick channel
            #####################
            # TRIAL-WISE FIGURE #
            #####################
            # Sort if needed
            fields_for_sorting = []
            for field in comparison['sort']:
                fields_for_sorting.append(epochs[query].metadata[field])
            if len(fields_for_sorting) == 1:
                mylist = [(i, j) for (i, j) in zip(range(len(fields_for_sorting[0])), fields_for_sorting[0])]
                IX = [i[0] for i in sorted(mylist, key=itemgetter(1))]
                mylist_sorted = sorted(mylist, key=itemgetter(1))
                yticklabels = [str(e[1]) for e in mylist_sorted]
            elif len(fields_for_sorting) == 2:
                mylist = [(i, j, k) for (i, j, k) in zip(range(len(fields_for_sorting[0])), fields_for_sorting[0], fields_for_sorting[1])]
                IX = [i[0] for i in sorted(mylist, key=itemgetter(1, 2))]
                mylist_sorted = sorted(mylist, key=itemgetter(1, 2))
                yticklabels = [str(e[1])+'-'+str(e[2]) for e in mylist_sorted]
            elif len(fields_for_sorting) == 3:
                mylist = [(i, j, k, l) for (i, j, k, l) in zip(range(len(fields_for_sorting[0])), fields_for_sorting[0], fields_for_sorting[1], fields_for_sorting[2])]
                IX = [i[0] for i in sorted(mylist, key=itemgetter(1, 2, 3))]
                mylist_sorted = sorted(mylist, key=itemgetter(1, 2, 3))
                yticklabels = [str(e[1])+'-'+str(e[2])+'-'+str(e[3]) for e in mylist_sorted]
            data_curr_query = data_curr_query[IX, :] # sort data
            # plot query data
            ax = plt.subplot2grid((nrows, ncols+1), (nums_trials_cumsum[i_query]+spacing*i_query, 0), rowspan=height_query_data, colspan=10) # add axis to main figure
            if args.data_type == 'spike':
                cmap = 'binary'
            else:
                cmap = 'RdBu_r'
            if args.data_type == 'spike' and args.filter == 'raw' and args.smooth_raster: # smooth raster a little bit
                num_trials = data_curr_query.shape[0]
                print(num_trials)
                data_curr_query_smoothed = data_curr_query.copy()
                for t in range(num_trials):
                    data_curr_query_smoothed[t, :] = gaussian_filter1d(data_curr_query[t, :], float(args.smooth_raster)*1000) # 1000Hz is assumed as sfreq
                #im = ax.imshow(data_curr_query_smoothed, interpolation='nearest', aspect='auto', vmin=args.vmin, vmax=args.vmax, cmap=cmap)
                print(data_curr_query_smoothed.shape[0])
                im = ax.imshow(data_curr_query_smoothed, cmap=cmap, interpolation='none', aspect='auto')
            else:
                im = ax.imshow(data_curr_query, interpolation='nearest', aspect='auto', cmap=cmap)
            ax.tick_params(axis='x', which='both', bottom='off', labelbottom='off')
            ax.set_xticks([])
            if comparison['sort']:
                ax.set_yticks(range(0, len(fields_for_sorting[0]), args.y_tick_step))

                #yticklabels = np.sort(fields_for_sorting[0])[::args.y_tick_step]
                yticklabels = yticklabels[::args.y_tick_step]
                if args.yticklabels_sortkey:
                    yticklabels = [l.split('-')[args.yticklabels_sortkey].capitalize() for l in yticklabels]
                ax.set_yticklabels(yticklabels, fontsize=args.yticklabels_fontsize)
            ax.set_ylabel(condition_name, fontsize=10, color=color)
            ax.axvline(x=0, color='k', ls='--', lw=1) 
            # TAKE MEAN FOR ERP FIGURE 
            if args.data_type == 'spike':
                # Gausssian smoothing of raster ERPs
                if args.level == 'phone':
                    gaussian_w = 0.002 # in sec
                else:
                    gaussian_w = 0.01 # in sec
                num_trials = data_curr_query.shape[0]
                for t in range(num_trials):
                    data_curr_query[t, :] = gaussian_filter1d(data_curr_query[t, :], gaussian_w * 1000) # 1000Hz is assumed as sfreq
                #print(np.max(data_curr_query))
                data_mean = np.mean(data_curr_query, axis=0)  
                data_mean = np.expand_dims(data_mean, axis=0)
                evoked_curr_query = mne.EvokedArray(data_mean, epochs[query].pick(ch_name).info, epochs.tmin, nave=num_trials)
            else:
                evoked_curr_query = epochs[query].pick(ch_name).average(method='median')
            evoked_curr_query.data = evoked_curr_query.data/1e3 # HACK: Revert auto scaling by MNE viz.plot_compare_evokeds
            evoked_dict[condition_name] = evoked_curr_query 
            if args.data_type != 'spike':
                if i_query == 0: # determine cmin cmax based on first query
                    perc10, perc90 = np.percentile(data_curr_query, 10), np.percentile(data_curr_query, 90)
                im.set_clim([perc10, perc90])
            else:
                #if i_query == 0: # determine cmin cmax based on first query
                #    max_val = 0.5*np.max(data_curr_query_smoothed)
                max_val = 0.1
                im.set_clim([0, max_val])
       
        ##############
        # ERP FIGURE #
        ##############
        
        if args.data_type == 'spike':
            label_y = 'firing rate (Hz)'
            ylim = [-1, 30]
            yticks = [0, 10, 20, 30]
        else:
            if args.filter == 'high-gamma':
                label_y = 'dB'
                ylim = [-1.5, 1.5]
                yticks = [-1.5, -1, 0, 1, 1.5]
            else:
                label_y = 'IQR-scale'
                ylim = [-3, 3]
                yticks = [-3, -1.96, 0, 1.96, 3]
        #fig_erp = mne.viz.plot_compare_evokeds(evoked_dict, show=False, colors=colors_dict, picks=ch_name, axes=ax2, ylim={'eeg':ylim}, title='')
        fig_erp = mne.viz.plot_compare_evokeds(evoked_dict, show=False, colors=colors_dict, linestyles=linestyles_dict, picks=ch_name, axes=ax2, title='')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc=2)
        ax2.set_ylabel(label_y, fontsize=16)
        ax2.set_ylim(ylim)
        ax2.set_yticks(yticks)

        #############
        # COLOR BAR #
        #############
        if args.data_type != 'spike':
            cbaxes = plt.subplot2grid((nrows, ncols+1), (0, 10), rowspan=sum(nums_trials), colspan=1) # cbar
            cbar = plt.colorbar(im, cax=cbaxes)
            if args.filter == 'high-gamma':
                label_cbar = 'dB'
            else:
                label_cbar = 'IQR-scale'
            cbar.set_label(label=label_cbar, size=22)

        if comparison['sort']:
            str_sort = 'Trials are sorted by:%s' % comparison['sort'][0]
        else:
            str_sort = ''
        # Add main title
        if not args.no_title:
            fig.suptitle('%s, %s\n%s\n%s' % (args.patient, ch_name, args.comparison_name, str_sort), fontsize=12)
        plt.subplots_adjust(left=0.25, right=0.85)
        ########
        # SAVE #
        ########
        plt.savefig(fname_fig)
        print('fig saved to: %s' % fname_fig)
        plt.close()
