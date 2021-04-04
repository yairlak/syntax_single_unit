import mne
import numpy as np
from functions import load_settings_params, stats
from functions.utils import probename2picks
import argparse, os
# Set current working directory to that of script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(description='Generate plots')
parser.add_argument('--patient', default='479_11', help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], default='micro', help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='word', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'high-gamma'], default='high-gamma', help='')
parser.add_argument('--query', default=None, help='Query epochs before stat computation')
parser.add_argument('--crop-pre', nargs=2, default=[], type=float, help='Pre-stimulus period for computation - (tmin, tmax). Should be entered with two args, e.g., --crop-pre -0.5 0')
parser.add_argument('--crop-post', nargs=2, default=(0, 1), type=float, help='Post-stimulus period for computation - (tmin, tmax). Should be entered with two args, e.g., --crop-pre 0.1 0.6')
parser.add_argument('--probe-name', default=[], nargs='*', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=[], nargs='*', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--threshold', default=None, type=float, help='Threshold for cluster permutation test')
parser.add_argument('--tail', default=1, type=int, help='tail of stat test')
parser.add_argument('--n-jobs', default=1, type=int, help='number of jobs for parallelization')
parser.add_argument('--n-permutation', default=1000, type=int, help='number of permutation for cluster test')
parser.add_argument('--extention', default='res', type=str, help='Extention of output file.')

#
args = parser.parse_args()
args.patient = 'patient_' + args.patient


print(args)
kwargs_dict = {'threshold':args.threshold,
               'tail':args.tail,
               'n_jobs':args.n_jobs,
               'n_permutations':args.n_permutation}
responsive_channels = []

# LOAD EPOCHS OBJECT
settings = load_settings_params.Settings(args.patient)
fname = '%s_%s_%s_%s-epo' % (args.patient, args.data_type, args.filter, args.level)
epochs = mne.read_epochs(os.path.join(settings.path2epoch_data, fname+'.fif'), preload=True)

fname_out = os.path.join(settings.path2epoch_data, fname + '.' + args.extention)

# PICK
if args.probe_name:
    print('-'*100)
    picks = probename2picks(args.probe_name, epochs.ch_names, args.data_type)
    print(picks)
    epochs.pick_channels(picks)
elif args.channel_name:
    epochs.pick_channels(args.channel_name)
elif args.channel_num:
    epochs.pick(args.channel_num)

# REMOVE MICROPHONE IF IN CHANNEL NAMES
picks = mne.pick_channels(epochs.ch_names, include=[], exclude=['MICROPHONE'])
epochs.pick(picks)

# QUERY
if args.query is not None:
    epochs = epochs[args.query]

# CROP
if args.crop_pre:
    epochs_pre = epochs.copy().crop(args.crop_pre[0], args.crop_pre[1])
else: # compare to zero activations if pre-period was not provided
    print('Pre-period was not provided -- running statistical test with respect to a zero baseline')
    epochs_pre = epochs.copy().crop(args.crop_post[0], args.crop_post[1])
    epochs_pre._data = np.zeros(epochs_pre.get_data().shape)
epochs_post = epochs.copy().crop(args.crop_post[0], args.crop_post[1])

# GET STATS
responsive_channels = stats.get_comparison_stats(epochs_post, epochs_pre, **kwargs_dict)

# WRITE TO TEXT FILE
with open(fname_out, 'w') as f:
    f.write(f"Epochs filename: {fname}.fif\n")
    f.write("Query: " + str(args.query) + '\n')
    f.write("Crop period (pre-stimulus): " + str(args.crop_pre) + '\n')
    f.write("Crop period (post-stimulus): " + str(args.crop_post)+ '\n')
    for l in responsive_channels:
        cluster_p_values = ';'.join(map(str, l['cluster_p_values']))
        clusters = ';'.join(['-'.join(map(str, (c[0].start, c[0].stop))) for c in l['clusters']])
        curr_line = f"{l['ch_IX']}, {l['ch_name']}, {cluster_p_values}, {clusters}\n" 
        f.write(curr_line)

print(f'Responsiveness file saved to: {fname_out}')
