# This script counts the number of cluster groups found in each channel
#
# EXAMPLE:
#  python count_spike_clusters.py --patient 479_11 --patient 487 --sign pos

import argparse, os, sys, glob
# Set current working directory to that of script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
import numpy as np
from pprint import pprint
from functions import load_settings_params, read_logs_and_features, convert_to_mne, data_manip, analyses_single_unit

parser = argparse.ArgumentParser(description='Generate MNE-py epochs object for a specific frequency band for all channels.')
parser.add_argument('--patient', action='append', help='Patient string (e.g., 479_11, 487')
parser.add_argument('--channel', action='append', default=[], help="Channels to analyze. If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--sign', action='append', default=[], help="Sign of spike polarity - pos/neg")
args = parser.parse_args()

args.patient = ['patient_' + p for p in args.patient]
if not args.sign:
    args.sign = ['pos', 'neg']

for patient in args.patient:
    print('-' * 100)
    print(' PATIENT %s ' % str(patient))
    print('-' * 100)
    settings = load_settings_params.Settings(patient)
    channels_with_spikes = data_manip.get_channels_with_spikes_from_combinato_sorted_h5(settings, args.sign)
    #for sign in args.sign:
    total_pos, total_neg = (0, 0)
    for sign in args.sign:
        print('-'*20)
        print('POSITIVE POLARITY')
        print('-'*20)
        for sublist in channels_with_spikes:
            if (sign=='pos') & (sublist[2] > 0):
                total_pos += sublist[2]
                print('Channel %i %s (pos): %i clusters' % (sublist[0], sublist[1], sublist[2]))
            if (sign=='neg') & (sublist[3] > 0):
                total_negs += sublist[3]
                print('Channel %i %s (neg): %i clusters' % (sublist[0], sublist[1], sublist[3]))
    if 'pos' in args.sign:
        print('Total number of cluster groups (pos): %i' % total_pos)
    if 'neg' in args.sign:
        print('Total number of cluster groups (neg): %i' % total_neg)
