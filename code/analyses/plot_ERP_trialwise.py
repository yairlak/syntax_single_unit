import argparse
import os
import pickle
import mne
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from plotting.viz_ERPs import get_sorting, average_repeated_trials
from utils import comparisons, load_settings_params
from utils.data_manip import DataHandler
from utils.utils import probename2picks, update_queries
from scipy.ndimage import gaussian_filter1d

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
                    default='sentence_onset', help='')
parser.add_argument('--filter', default='raw', help='')
parser.add_argument('--smooth', default=None, help='')
parser.add_argument('--scale-epochs', action="store_true", default=False, help='')
# PICK CHANNELS
parser.add_argument('--probe-name', default=[], nargs='*', type=str,
                    help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=['LSTG7_14_p2'], nargs='*', type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=None, nargs='*', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
# QUERY (SELECT TRIALS)
parser.add_argument('--comparison-name', default='479_11_LSTG7_15p2_phone', help='int. Comparison name from Code/Main/functions/comparisons_level.py. see print_comparisons.py')
parser.add_argument('--block-type', default=[], help='Block type will be added to the query in the comparison')
parser.add_argument('--fixed-constraint', default=[], help='A fixed constrained added to query. For example first_phone == 1 for auditory blocks')
parser.add_argument('--average-repeated-trials', action="store_true", default=False, help='')
parser.add_argument('--tmin', default=-0.1, type=float, help='crop window. If empty, only crops 0.1s from both sides, due to edge effects.')
parser.add_argument('--tmax', default=0.6, type=float, help='crop window')
parser.add_argument('--baseline', default=None, type=str, help='Baseline to apply as in mne: (a, b), (None, b), (a, None), (None, None) or None')
parser.add_argument('--baseline-mode',  default=None, choices=['mean', 'ratio', 'logratio', 'percent', 'zscore', 'zlogratio'], help='Type of baseline method')
# MISC
parser.add_argument('--SOA', default=500, help='SOA in design [msec]')
parser.add_argument('--word-ON-duration', default=250, help='Duration for which word word presented in the RSVP [msec]')
parser.add_argument('--remove-outliers', action="store_true", default=False, help='Remove outliers based on percentile 25 and 75')
parser.add_argument('--no-title', action="store_true", default=True)
parser.add_argument('--yticklabels-sortkey', type=int, default=[], help="")
parser.add_argument('--yticklabels-fontsize', type=int, default=20, help="")
parser.add_argument('--dont-write', default=False, action='store_true', help="If True then file will be overwritten")
parser.add_argument('--sort-key', default=['word_string'], help='Keys to sort according')
parser.add_argument('--y-tick-step', default=100, type=int, help='If sorted by key, set the yticklabels density')
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

# LOAD
data = DataHandler(args.patient, args.data_type, args.filter,
                   args.probe_name, args.channel_name, args.channel_num)
# Both neural and feature data into a single raw object
data.load_raw_data(verbose=True)
print(data.raws[0].info['sfreq'])

# COMPARISON
comparisons = comparisons.comparison_list()
comparison = comparisons[args.comparison_name].copy()


if 'level' in comparison.keys():
    args.level = comparison['level']

# GET SENTENCE-LEVEL DATA BEFORE SPLIT
data.epoch_data(level=args.level,
                query=None,
                scale_epochs=args.scale_epochs,
                smooth=args.smooth,
                verbose=True)
epochs = data.epochs[0]

print(epochs.metadata['semantic_categories_names'])

if 'sort' not in comparison.keys():
    comparison['sort'] = args.sort_key
else:
    print(comparison['sort'])

if 'tmin_tmax' in comparison.keys():
    args.tmin, args.tmax = comparison['tmin_tmax']
if 'y-tick-step' in comparison.keys():
    args.y_tick_step = comparison['y-tick-step']
if 'fixed_constraint' in comparison.keys():
    args.fixed_constraint = comparison['fixed_constraint']

comparison = update_queries(comparison, args.block_type,
                            args.fixed_constraint, epochs.metadata)

if isinstance(args.y_tick_step, int):
    args.y_tick_step = [args.y_tick_step] * len(comparison['queries'])

print(args.comparison_name)
pprint(comparison)

# PICK
# if args.probe_name:
#     picks = probename2picks(args.probe_name, epochs.ch_names, args.data_type)
#     epochs.pick_channels(picks)
# elif args.channel_name:
#     epochs.pick_channels(args.channel_name)
# elif args.channel_num:
#     epochs.pick(args.channel_num)

# # Filter non-responsive channels
# if args.responsive_channels_only:
#     if args.data_type == 'spike':
#         filt = 'gaussian-kernel' # if spikes, plot raw but calc signficance for responsivness based on smoothed data.
#     else:
#         filt = args.filter
#     picks = pick_responsive_channels(epochs.ch_names, args.patient, args.data_type, filt, [args.block_type], p_value=0.01)
#     if picks:
#         epochs.pick_channels(picks)
#     else:
#         raise('No responsive channels were found')


print('-'*100)
print(sorted(epochs.ch_names))

# BASELINE
if args.filter != 'high-gamma': # high-gamma is already baselined during epoching (generate_epochs.py)
    if args.baseline:
        print('Apply baseline:')
        epochs.apply_baseline(args.baseline, verbose=True)
else: #baseline high-gamma (e.g., to dB)
    # pass
    if args.baseline and args.baseline_mode:
        epochs._data = mne.baseline.rescale(epochs.get_data(), epochs.times, args.baseline, args.baseline_mode) 
#print(epochs._data[:2, :100])

# CROP
if args.tmin and args.tmax:
    epochs.crop(args.tmin, args.tmax)
#else:
    #if args.filter == 'high-gamma': # remove boundary effects
    #    epochs.crop(min(epochs.times) + 0.1, max(epochs.times) - 0.1)

for ch, ch_name in enumerate(epochs.ch_names):
    print(ch_name)
    #if ch_name == 'MICROPHONE': continue
    # output filename of figure
    str_comparison = '_'.join([tup[0] for tup in comparison['queries']])
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

    if (not os.path.exists(fname_fig)) or (not args.dont_write): # Check if output fig file already exists: 
        # Get number of trials from each query
        nums_trials = []; ims = []
        for query in comparison['queries']:
            data_curr_query = epochs[query].pick(ch_name).get_data()[:, 0, :]
            if args.average_repeated_trials:
                _, yticklabels, _ = get_sorting(epochs,
                                                query,
                                                comparison['sort'],
                                                ch_name, args)
                
                    
                data_curr_query, yticklabels = average_repeated_trials(data_curr_query, yticklabels)
            nums_trials.append(data_curr_query.shape[0]) # query and pick channel
        print('Number of trials from each query:', nums_trials)
        nums_trials_cumsum = np.cumsum(nums_trials)
        nums_trials_cumsum = [0] + nums_trials_cumsum.tolist()
        # Prepare subplots
        if 'figsize' in comparison.keys():
            figsize = comparison['figsize']
        else:
            if args.level == 'word':
                figsize = (4, 20)
            else:
                figsize = (15, 10)
        if args.level == 'word':
            fig, axs = plt.subplots(len(nums_trials)+1, 1,figsize=figsize, gridspec_kw={'height_ratios':nums_trials + [int(sum(nums_trials)/3)]})
            num_queries = len(comparison['queries'])
            #height_ERP = int(np.ceil(sum(nums_trials)/num_queries))
            height_ERP = np.max(nums_trials)
        else:
            fig, axs = plt.subplots(len(nums_trials)+1, 1,figsize=figsize, gridspec_kw={'height_ratios':nums_trials + [int(sum(nums_trials)/3)]})
            num_queries = len(comparison['queries'])
            height_ERP = int(np.ceil(sum(nums_trials)/num_queries))
        if num_queries > 1:
            spacing = int(np.ceil(0.1*sum(nums_trials)/num_queries))
        else:
            spacing = 0
        nrows = sum(nums_trials)+height_ERP+spacing*num_queries; ncols = 10 # number of rows in subplot grid per query. Width is set to 10. num_queries is added for 1-row spacing
        # prepare axis for ERPs 
        # ax2 = plt.subplot2grid((nrows, ncols+1), (sum(nums_trials)+spacing*num_queries, 0), rowspan=height_ERP, colspan=10) # Bottom figure for ERP
        # Collect data from all queries and sort based on args.sort_key
        data = []
        evoked_dict = dict()
        colors_dict = {}
        linestyles_dict = {}
        styles = {}
        for i_query, query in enumerate(comparison['queries']):
            condition_name = comparison['condition_names'][i_query]
            height_query_data = nums_trials[i_query]
            color = comparison['colors'][i_query]
            colors_dict[condition_name] = color
            if 'ls' in comparison.keys():
                ls = comparison['ls'][i_query]
                linestyles_dict[condition_name] = ls
            if 'lw' in comparison.keys():
                lw = comparison['lw'][i_query]
                styles[condition_name] = {"linewidth":lw}
            else:
                lw = 8 # default linewidth
                styles[condition_name] = {"linewidth":lw}

            data_curr_query = epochs[query].pick(ch_name).get_data()[:, 0, :] # query and pick channel
            # if args.baseline and args.data_type == 'high-gamma':
            #     data_curr_query = mne.baseline.rescale(data_curr_query,
            #                                            epochs.times,
            #                                            args.baseline,
            #                                            args.baseline_mode)
            #####################
            # TRIAL-WISE FIGURE #
            #####################
            
            word_strings = epochs[query].metadata['word_string']
            IX, yticklabels, fields_for_sorting = get_sorting(epochs,
                                                              query,
                                                              comparison['sort'],
                                                              ch_name, args)
            if 'yticklabels' in comparison.keys():
                yticklabels = epochs[query].metadata[comparison['yticklabels']].to_numpy()[IX]
            data_curr_query = data_curr_query[IX, :] # sort data
            
            if args.average_repeated_trials:
                data_curr_query, yticklabels = average_repeated_trials(data_curr_query, yticklabels)
                data_curr_query = data_curr_query[::-1, :]
                yticklabels = np.asarray(yticklabels)[::-1]
                    
                    
            # plot query data
            # ax = plt.subplot2grid((nrows, ncols+1), (nums_trials_cumsum[i_query]+spacing*(i_query+1), 0), rowspan=height_query_data, colspan=10) # add axis to main figure
            if args.data_type == 'spike':
                if 'cmaps' in comparison.keys():
                    cmap = comparison['cmaps'][i_query]
                else:
                    cmap = 'binary'
            else:
                cmap = 'RdBu_r'
            if args.data_type == 'spike' and args.filter == 'raw':
                num_trials = data_curr_query.shape[0]
                data_curr_query_smoothed = data_curr_query.copy()
                if args.smooth_raster: # smooth raster a little bit
                    for t in range(num_trials):
                        data_curr_query_smoothed[t, :] = gaussian_filter1d(data_curr_query[t, :], float(args.smooth_raster)*1000) # 1000Hz is assumed as sfreq
                #im = ax.imshow(data_curr_query_smoothed, interpolation='nearest', aspect='auto', vmin=args.vmin, vmax=args.vmax, cmap=cmap)
                #print(data_curr_query_smoothed.shape[0])
                im = axs[i_query].imshow(data_curr_query_smoothed, cmap=cmap, interpolation='none', aspect='auto')
                # axs[i_query].set_facecolor('r')
                # axs[i_query].patch.set_alpha(0.5)
            else:
                im = axs[i_query].imshow(data_curr_query, interpolation='nearest', aspect='auto', cmap=cmap)
            axs[i_query].tick_params(axis='x', which='both',
                                     bottom='off', labelbottom='off')
            axs[i_query].set_xticks([])
            if color is not None:
                axs[i_query].tick_params(axis='y', colors=color)
            if isinstance(comparison['sort'], list):
                axs[i_query].set_yticks(range(0, len(fields_for_sorting[0]), args.y_tick_step[i_query]))

                #yticklabels = np.sort(fields_for_sorting[0])[::args.y_tick_step]
                yticklabels = yticklabels[::args.y_tick_step[i_query]]
                if args.yticklabels_sortkey:
                    yticklabels = [l.split('-')[args.yticklabels_sortkey].capitalize() for l in yticklabels]
                axs[i_query].set_yticklabels(yticklabels, fontsize=args.yticklabels_fontsize)
            elif comparison['sort'] == 'clustering':
                axs[i_query].set_yticks(range(0, len(yticklabels), args.y_tick_step[i_query]))
                yticklabels = yticklabels[::args.y_tick_step]
                axs[i_query].set_yticklabels(yticklabels, fontsize=args.yticklabels_fontsize)
            # ax.set_ylabel(condition_name, fontsize=10, color=color, rotation=0, labelpad=20)
            timezero = list(epochs.times).index(0)
            axs[i_query].axvline(x=timezero, color='k', ls='--', lw=3) 
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
                # print(np.max(data_curr_query))
                data_mean = np.mean(data_curr_query, axis=0)  
                data_mean = np.expand_dims(data_mean, axis=0)
                evoked_curr_query = mne.EvokedArray(data_mean, epochs[query].pick(ch_name).info, epochs.tmin, nave=num_trials)
            else:
                print(epochs.ch_names, query, ch_name)
                evoked_curr_query = epochs[query].pick(ch_name).average(method='median')
            # if args.data_type != 'spike':
            ch_type = epochs.get_channel_types(picks=[ch])[0]
            #print(ch_type)
            if ch_type == 'seeg' and args.data_type != 'spike': # HACK: Revert auto scaling by MNE viz.plot_compare_evokeds
                evoked_curr_query.data = evoked_curr_query.data/1e3
            elif ch_type == 'eeg':
                evoked_curr_query.data = evoked_curr_query.data/1e3 
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
            # label_y = 'firing rate (Hz)'
            label_y = ''
            if 'ylim' in comparison.keys():
                ymax = comparison['ylim']
            else:
                ymax = 40
            ylim = (-1, ymax)
            yticks = range(0, ymax, 20)
        else:
            if args.filter == 'high-gamma':
                label_y = ''
                ylim = [-1, 1]
                yticks = [ylim[0], 0, ylim[1]]
            else:
                label_y = 'IQR-scale'
                ylim = [-3, 3]
                yticks = [-3, -1.96, 0, 1.96, 3]
        #fig_erp = mne.viz.plot_compare_evokeds(evoked_dict, show=False, colors=colors_dict, picks=ch_name, axes=ax2, ylim={'eeg':ylim}, title='')
        fig_erp = mne.viz.plot_compare_evokeds(evoked_dict, show=False,
                                               colors=colors_dict,
                                               linestyles=linestyles_dict,
                                               picks=ch_name,
                                               axes=axs[-1], title='')
        # ax2.legend(bbox_to_anchor=(1.05, 1), loc=2)
        axs[-1].legend().set_visible(False)
        axs[-1].set_ylabel(label_y, fontsize=16, rotation=0, labelpad=20)
        axs[-1].set_xlabel('') # Remove xlabel
        axs[-1].tick_params(axis='both', labelsize=20)
        axs[-1].set_ylim(ylim)
        axs[-1].set_yticks(yticks)
        if 'tmin_tmax' in comparison.keys():
            xticks = [0, comparison['tmin_tmax'][1]]
            axs[-1].set_xticks(xticks)

        #############
        # COLOR BAR #
        #############
        if args.data_type != 'spike':
            fig_cbar, ax_cbar = plt.subplots(1,1,figsize=(2,2))
            # cbaxes = plt.subplot2grid((nrows, ncols+1), (0, 10), rowspan=int(sum(nums_trials)/10), colspan=1) # cbar
            cbar = plt.colorbar(im, cax=ax_cbar)
            if args.filter == 'high-gamma':
                label_cbar = 'dB'
            else:
                label_cbar = 'IQR-scale'
            cbar.set_label(label=label_cbar, size=22)
            plt.tight_layout()
            plt.savefig(fname_fig[:-4]+'_cbar.png')
            print('fig saved to: %s' % fname_fig[:-4]+'_cbar.png')
            plt.close(fig_cbar)

        if comparison['sort']:
            str_sort = 'Trials are sorted by:%s' % comparison['sort'][0]
        else:
            str_sort = ''
        # Add main title
        if not args.no_title:
            fig.suptitle('%s, %s\n%s\n%s' % (args.patient, ch_name, args.comparison_name, str_sort), fontsize=12)
        plt.subplots_adjust(left=0.3)
        ########
        # SAVE #
        ########
        plt.tight_layout()
        plt.savefig(fname_fig, facecolor=fig.get_facecolor(), edgecolor='none')
        print('fig saved to: %s' % fname_fig)
        plt.close()
