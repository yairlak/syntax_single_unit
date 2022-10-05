#  Generate raw mne files (fif formant) from mat or combinato files:
# - In the case of data-type = 'macro' bi-polar referencing is applied.
# - Notch filtering of line noise is performed.
# - clipping using robustScalar transform is applied
#   by using -5 and 5 for lower/upper bounds.
# - The output is a raw mne object saved to Data/UCLA/patient_?/Raw/

import os
import argparse
from utils import data_manip
import numpy as np
import scipy.io as sio

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser()
parser.add_argument('--patient', default='556', help='Patient number')
parser.add_argument('--data-type',
                    choices=['micro', 'macro', 'spike', 'microphone'],
                    default='macro', help='macro/micro/spike')
parser.add_argument('--verbose', '-v', action='store_true',
                    default=False)
args = parser.parse_args()
args.patient = 'patient_' + args.patient
print(args)

path2rawdata = os.path.join('..', '..', 'Data', 'UCLA',
                            args.patient, 'Raw', args.data_type)

data, ch_ids, ch_ids_hdr, sfreq = data_manip.ns2mat(args.data_type, path2rawdata)

if args.verbose:
    print(f'Sampling rate: {sfreq}')
    print(ch_ids)
    print(ch_ids_hdr)
    print(f'Shape of data: {data[0].shape}')

assert len(ch_ids) == data[0].shape[1]

path2output = os.path.join(path2rawdata, 'mat')
os.makedirs(path2output, exist_ok=True)

for ch_id, i_ch in zip(ch_ids_hdr, range(data[0].shape[1])):
    fn_mat = os.path.join(path2output, f'CSC{ch_id}.mat')

    curr_data = data[0][:, i_ch]
    print(curr_data.mean(), curr_data.std())
    sio.savemat(fn_mat, {'data':curr_data,
                         'ch_id':ch_id,
                         'sr':sfreq})
    print(f'Mat file saved to: {fn_mat}')
