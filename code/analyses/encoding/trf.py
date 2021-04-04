import argparse, os, sys, pickle
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
import mne
from functions import comparisons
from functions.utils import dict2filename, update_queries
from functions.data_manip import load_epochs_data
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from mne.decoding import ReceptiveField, TimeDelayingRidge
import scipy.io as sio


parser = argparse.ArgumentParser(description='Generate plots for TIMIT experiment')
# DATA
parser.add_argument('--patient', action='append', default=[], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], action='append', default=[], help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='word', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'high-gamma'], action='append', default=[], help='')
parser.add_argument('--probe-name', default=[], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
# QUERY
parser.add_argument('--comparison-name', default='all_trials', help='Comparison name from Code/Main/functions/comparisons.py')
parser.add_argument('--block-type', choices=['auditory', 'visual'], default='auditory', help='Block type will be added to the query in the comparison')
parser.add_argument('--fixed-constraint', default=[], help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--label-from-metadata', default=[], help='Field name in metadata that will be used to generate labels for the different classes. If empty, condition_names in comparison will be used')
# FEATURES
parser.add_argument('--path2features', default=[], help="Path to filename with features for all trials. Extension could be either mat (Matlab) or csv (from e.g. ndarray).")
# MISC
parser.add_argument('--vmin', default=None, type=float, help='')
parser.add_argument('--vmax', default=None, type=float, help='')
parser.add_argument('--times', nargs='*', default=[0.1], type=float, help='')
parser.add_argument('--time-window', default=0.1, type=float, help='')
parser.add_argument('--decimate', default=[], type=float, help='If not empty, (for speed) decimate data by the provided factor.')
parser.add_argument('--path2figures', default=[], help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--dont-overwrite', default=False, action='store_true', help="If True then file will be overwritten")

# PARSE
args = parser.parse_args()
args.patient = ['patient_' + p for p in  args.patient]
print(mne.__version__)

# Which args to have in fig filename
list_args2fname = ['patient', 'data_type', 'filter', 'level', 'comparison_name', 'block_type', 'channel_name']
if args.probe_name: list_args2fname.append('probe_name')
if args.responsive_channels_only: list_args2fname += ['responsive_channels_only']
print(list_args2fname)

if not args.path2figures:
    args.path2figures = os.path.join('..', '..', 'Figures', 'TRFs')


########
# DATA #
########
epochs_list = load_epochs_data(args)
print('-'*100, '\n INPUT ARGUMENTS:')
pprint(args)
print('-'*100, '\n CHANNELS PER PATIENT:')
[print(f"{p}: {e.ch_names}") for p, e in zip(args.patient, epochs_list)]

###########
# Queries #
###########

# COMPARISON
comparisons = comparisons.comparison_list()
comparison = comparisons[args.comparison_name].copy()
comparison = update_queries(comparison, args.block_type, args.fixed_constraint, epochs_list[0].metadata)
print('-'*100, '\n COMPARISON:')
pprint(comparison)



#############################
# LOAD AUDITORY SPECTOGRAMS #
#############################
# Load data from features dict which contains a list of ndarray with len(list)=#epochs. Each array has size: n_features X n_timeframes. 
# The dict contains also the sfreq.
dict_features = pickle.load(open(args.path2features, 'rb'))
list_feature_mat = dict_features['specs'] # len(list)=n_epochs

# If acoustic features, duplicate three 3 times for all 3 blocks:
# print(sentence_numbers, set(sentence_numbers), len(sentence_numbers))
sentence_numbers = epochs_list[0][comparison['queries'][0]].metadata['sentence_number']
list_feature_mat += [list_feature_mat[IX-1] for IX in sentence_numbers[152:304]]
list_feature_mat += [list_feature_mat[IX-1] for IX in sentence_numbers[304:456]]
X_features = np.concatenate(list_feature_mat, axis = 1) # 
print('len list feature mat', len(list_feature_mat))

list_neural_response = []
for i_trial, feature_mat in enumerate(list_feature_mat): #loop over trials
    ################################
    # PREPARE NEURAL-RESPONSE DATA #
    ################################
    num_frames = feature_mat.shape[1]
    for epochs in epochs_list: # loop over epochs from different patients or probes
        curr_epochs = epochs[comparison['queries'][0]].copy().resample(dict_features['sfreq']).crop(tmin=0) # query, decimate (to bring neural sfreq to that of feature) and crop based on feature mat.
        y = curr_epochs[i_trial].get_data()
        list_neural_response.append(y[0, :, :num_frames])
    print('sfreq (feature mat, neural data): ', dict_features['sfreq'], epochs.info['sfreq'])

y = np.concatenate(list_neural_response, axis=1) # cat different patients/probes as new channel features
print('Feature tensor (n_features (freqs) X n_timepoints)', X_features.shape)
print('Neural response (n_channels X n_timepoints):', y.shape)


#######
# TRF #
#######
X_features = X_features.swapaxes(0, 1)
y = y.swapaxes(0, 1)
alphas = np.logspace(-3, 3, 7)
scores_lap = np.zeros_like(alphas)
models_lap = []
for ii, alpha in enumerate(alphas):
    estimator = TimeDelayingRidge(-0.1, 0.5, dict_features['sfreq'], reg_type='laplacian', alpha=alpha)
    rf = ReceptiveField(-0.1, 0.5, dict_features['sfreq'], feature_names = dict_features['freqs'], estimator=estimator)
    rf.fit(X_features, y)

    # Now make predictions about the model output, given input stimuli.
    #scores_lap[ii] = rf.score(X_test, y_test)
    models_lap.append(rf)

#ix_best_alpha_lap = np.argmax(scores_lap)

#kwargs = dict(vmax=np.abs(weights).max(), vmin=-np.abs(weights).max(), cmap='RdBu_r', shading='gouraud')
kwargs = dict(cmap='RdBu_r', shading='gouraud')
# Plot the STRF of each ridge parameter
fig, axs = plt.subplots(1, len(alphas), figsize=(20, 5))
times = rf.delays_ / float(rf.sfreq)
xlim = times[[0, -1]]
for ii, (rf_lap, i_alpha) in enumerate(zip(models_lap, alphas)):
    try:
        axs[ii].pcolormesh(times, rf_lap.feature_names, rf_lap.coef_[0], cmap='RdBu_r')#, **kwargs)
        axs[ii].set(xticks=dict_features['time_bins'][::100], xticklabels=dict_features['time_bins'][::100], yticks=dict_features['freqs'], yticklabels=dict_features['freqs'], xlim=xlim)
        if ii == 0:
            axs[ii].set(ylabel='Laplacian')
    except:
        print(times.shape, rf_lap.feature_names, rf_lap.coef_.shape)
        raise()
fig.suptitle('Model coefficients / scores for laplacian regularization', y=1)
mne.viz.tight_layout()


fname_fig = dict2filename(args.__dict__, '_', list_args2fname, 'png', True)
fname_fig = os.path.join(args.path2figures, 'TRF_' + fname_fig)
fig.savefig(fname_fig)
print('Figures saved to: ' + fname_fig)

