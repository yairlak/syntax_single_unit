# This script generates raw mne files (fif formant) from mat or combinato files:
# - In the case of data-type = 'macro' bi-polar referencing is applied. 
# - Notch filtering of line noise is performed.
# - clipping using robustScalar transform is applied (but data is *not* scaled at this stage), by using -3 and 3 for lower/upper bounds.
# - In the case of filter = 'gaussian-kernel', smoothing is applied before saving.
# - The output is a raw mne object saved to Data/UCLA/patient_?/Raw/

import os, argparse, re, sys, glob
# Set current working directory to that of script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
#sys.path.append('..')
from functions import load_settings_params, read_logs_and_features, convert_to_mne, data_manip, analyses
import numpy as np
from pprint import pprint
from sklearn.preprocessing import RobustScaler
from scipy.io import savemat

parser = argparse.ArgumentParser()
parser.add_argument('--patient', default='479_11', help='Patient number')
parser.add_argument('--data-type', choices = ['micro', 'macro', 'spike'], default='micro', help='macro/micro/spike')
parser.add_argument('--filter', choices = ['raw', 'gaussian-kernel'], default='raw', help='raw/gaussian-kernel. high-gamma is extrated by a separate Matlab script.')
parser.add_argument('--sfreq-downsample', default=1000, help='Downsampling frequency')
args = parser.parse_args()
args.patient = 'patient_' + args.patient
print(args)

print('Loading settings, params and preferences...')
settings = load_settings_params.Settings(args.patient)
params = load_settings_params.Params(args.patient)
pprint(settings.__dict__); pprint(params.__dict__)

# PATHS
if args.data_type == 'micro' or args.data_type == 'spike':
    path2CSC_mat = os.path.join(settings.path2rawdata, 'micro', 'CSC_mat')
elif args.data_type == 'macro':
    path2CSC_mat = os.path.join(settings.path2rawdata, 'macro', 'CSC_mat')

# GET CHANNALS AND PROBE NAMES
with open(os.path.join(path2CSC_mat, 'channel_numbers_to_names.txt')) as f_channel_names:
    channel_names = f_channel_names.readlines()

path2CSC_mat = os.path.join(path2CSC_mat, 'clean')
if not os.path.exists(path2CSC_mat): os.makedirs(path2CSC_mat)

channel_names_dict = dict(zip(map(int, [s.strip('\n').split('\t')[0] for s in channel_names]), [s.strip('\n').split('\t')[1][:-4] for s in channel_names]))
channel_nums = list(channel_names_dict.keys())
if args.data_type == 'micro':
    channel_nums = list(set(channel_nums) - set([0])) # REMOVE channel 0 (MICROPHONE)
    #channel_nums = list(set(channel_nums + [0])) # ADD channel 0 (MICROPHONE)
    channel_names_dict[0] = 'MICROPHONE'
else:
    if 0 in channel_nums:
        channel_nums = list(set(channel_nums) - set([0])) # REMOVE channel 0 (MICROPHONE)
        del channel_names_dict[0]

channel_nums.sort()
print('Number of channel %i: %s' % (len(channel_names_dict.values()), channel_names_dict.values()))


#channel_nums = [1, 2] # for DEBUG
for channel_num in channel_nums:
    channel_name = channel_names_dict[channel_num]
    if channel_num == 0:
        probe_name = 'MIC'
    else:
        if args.data_type=='macro': 
            probe_name = re.split('(\d+)', channel_name)[0]
        else:
            probe_name = re.split('(\d+)', channel_name)[2][1::]
    print('Current channel: %s (%i)' % (channel_name, channel_num))
    
    # LOAD DATA -> RAW OBJECT
    curr_raw = data_manip.load_channel_data(args.data_type, args.filter, channel_num, channel_name, probe_name, settings, params)
    ############################
    # Robust Scaling Transform #
    ############################
    curr_data = curr_raw.copy().get_data()
    assert curr_data.shape[0] == 1
    # curr_data = curr_data[0:1000, :] # FOR DEBUG (!)
    print('Fit scaler model')
    transformer = RobustScaler().fit(np.transpose(curr_data))
    print('Scale data')
    data_scaled = np.transpose(transformer.transform(np.transpose(curr_data)))

    ############
    # CLIPPING #
    ############
    lower, upper = -3, 3
    curr_data[data_scaled>upper] = transformer.inverse_transform([[upper],])[0][0]
    curr_data[data_scaled<lower] = transformer.inverse_transform([[lower],])[0][0]

    ########
    # SAVE #
    ########
    dict_data = {}
    dict_data['data'] = curr_data
    dict_data['elec_name'] = channel_name
    dict_data['samplingInterval'] = 1/curr_raw.info['sfreq']
    dict_data['sr'] = curr_raw.info['sfreq']
    print(dict_data, dict_data['data'].shape)
    fname = os.path.join(path2CSC_mat, 'CSC%i.mat' % (channel_num))
    savemat(fname, dict_data)
    print('Cleaned CSC mat file was saved to: %s' % fname)

