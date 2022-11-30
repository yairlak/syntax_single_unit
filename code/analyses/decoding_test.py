# -*- coding: utf-8 -*-

import argparse, os, pickle
import numpy as np
from scipy import stats
from decoding.utils import get_args2fname, update_args, get_comparisons
from decoding.data_manip import prepare_data_for_classification
from decoding.models import define_model
from decoding.data_manip import get_data
from sklearn.model_selection import LeaveOneOut, KFold
from utils.utils import dict2filename
from utils.utils import update_queries
from pprint import pprint
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from decoding.decoder import decode_comparison
from utils.utils import read_MNI_coord_from_xls
import pandas as pd

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(description='')
# DATA
parser.add_argument('--patient', action='append', default=[],
                    help='Patient number')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'],
                    action='append', default=[])
parser.add_argument('--level',
                    choices=['sentence_onset','sentence_offset',
                             'word', 'phone'],
                    default='sentence_onset')
parser.add_argument('--filter', choices=['raw', 'high-gamma'],
                    action='append', default=[], help='')
parser.add_argument('--probe-name', default=[], nargs='*', action='append',
                    type=str, help='e.g., LSTG, overrides channel_name/num')
parser.add_argument('--ROIs', default=None, nargs='*', type=str,
                    help='e.g., Brodmann.22-lh, overrides probe_name')
parser.add_argument('--channel-name', default=[], nargs='*', action='append',
                    type=str, help='e.g., GA1-LAH1')
parser.add_argument('--channel-num', default=[], nargs='*', action='append',
                    type=int, help='e.g., 3 to pick the third channel')
parser.add_argument('--responsive-channels-only', action='store_true',
                    default=False, help='Based on aud and vis files in Epochs folder')
parser.add_argument('--data-type_filters',
                    choices=['micro_high-gamma','macro_high-gamma',
                             'micro_raw','macro_raw', 'spike_raw'], nargs='*',
                             default=[], help='Only if args.ROIs is used')
parser.add_argument('--smooth', default=None, type=int,
                    help='gaussian width in [msec]')
# QUERY
parser.add_argument('--comparison-name', default=None,
                    help='See Utils/comparisons.py')
parser.add_argument('--comparison-name-test', default=None,
                    help='See Utils/comparisons.py')
parser.add_argument('--block-train', choices=['auditory', 'visual'],
                    default='auditory',
                    help='Block type is added to the query in the comparison')
parser.add_argument('--block-test', choices=['auditory', 'visual'],
                    default=None,
                    help='Block type is added to the query in the comparison')
parser.add_argument('--fixed-constraint', default=None,
                    help='e.g., "and first_phone == 1"')
parser.add_argument('--fixed-constraint-test', default=None,
                    help='e.g., "and first_phone == 1"')
parser.add_argument('--min-trials', default=10, type=float,
                    help='Minimum number of trials from each class.')
# DECODER
parser.add_argument('--classifier', default='logistic',
                    choices=['svc', 'logistic', 'ridge'])
parser.add_argument('--equalize-classes', default='downsample',
                    choices=['upsample', 'downsample'])
parser.add_argument('--gat', default=False, action='store_true',
                    help='If True, GAT will be computed; else, diagonal only')
# MISC
parser.add_argument('--coords', default=None, type=float, nargs='*',
                    help="coordinates (e.g., MNI) for searchlight")
parser.add_argument('--side', default=8, type=float, help='Side of cube in mm')
parser.add_argument('--tmin', default=None, type=float)
parser.add_argument('--tmax', default=None, type=float)
#parser.add_argument('--vmin', default=None, type=float, help='')
#parser.add_argument('--vmax', default=None, type=float, help='')
parser.add_argument('--decimate', default=50, type=int)
parser.add_argument('--cat-k-timepoints', type=int, default=1,
                    help='How many time points to concatenate before classification')
parser.add_argument('--path2figures', default='../../Figures/Decoding')
parser.add_argument('--path2output', default='../../Output/decoding')
parser.add_argument('--dont-overwrite', default=False, action='store_true',
                    help="If True then will not regenerate already existing figures")

args = parser.parse_args()
# CHECK AND UPDATE USER ARGUMENTS
args = update_args(args)

print('\nARGUMENTS:')
pprint(args.__dict__, width=1)

print('\nLOADING DATA:')
args.smooth = None

patients = '479 482 499 502 505 510 513 515 530 538 539 540 541 543 544 545 549 551 552 553 554 556'.split()
#patients = ['530']
path2data = os.path.join('..', '..', 'Data', 'UCLA', 'MNI_coords')
show_labels = False


def get_all_micro_electrodes(dict_row):
    patient = str(dict_row['patient'])
    if patient == '479':
        patient = '479_11'
    if patient == '554':
        patient = '554_13'
    path2fn = f'../../Data/UCLA/patient_{patient}/Raw/micro/channel_numbers_to_names.txt'
    df_nums2names = pd.read_csv(path2fn,
                                names=['ch_num', 'ch_name'],
                                delim_whitespace=True)

    # PICK
    probe_name = dict_row['probe_name']
    probe_names = [probe_name+str(i) for i in range(1, 9)]
    df_nums2names = df_nums2names[df_nums2names['ch_name'].str.contains(f"{'|'.join(probe_names)}")]

    return df_nums2names

    

# LOAD COORDINATES
path2coords = '../../Data/UCLA/MNI_coords/'
fn_coords = 'electrode_locations.csv'
df_coords_all = pd.read_csv(os.path.join(path2coords, fn_coords))
labels, coords, colors = [], [], []
for patient in patients:
    fn_coord = f'sub-{patient}_space-MNI_electrodes.xlsx'
    df_coords_patient = df_coords_all.query(f'patient=={patient}')
    for data_type in ['micro', 'macro']:
        #df_coords = read_MNI_coord_from_xls(os.path.join(path2data, fn_coord),
        #                                    data_type)
        if data_type == 'macro':
            df_coords = df_coords_patient.query('ch_type!="micro"')
            df_coords = df_coords.query('ch_type!="stim"')
        elif data_type == 'micro':
            df_coords = df_coords_patient.query('ch_type!="macro"')
            df_coords = df_coords.query('ch_type!="stim"')
            df_coords = df_coords[df_coords["electrode"].str.contains("_micro-1")]
            dfs_micro = []
            for i_row, row in df_coords.iterrows():
                df_micro = get_all_micro_electrodes(row)
                dfs_micro.append(df_micro)
            df_coords = pd.concat(dfs_micro)
        
        print(df_coords)

        curr_labels = list(df_coords['ch_name'].values)
        #curr_labels = [l.replace('-', '') for l in curr_labels]
        print('-'*100)
        print(patient, data_type)
        print('-'*100)
        print(f'Trying to pick {len(curr_labels)} channels')

        #continue
        if patient == '479':
            patient = '479_11'
        if patient == '554':
            patient = '554_13'
        args.patient = ['patient_' + patient]
        args.data_type = [data_type]
        args.filter = ['raw']
        args.channel_name = None
        data = get_data(args)
        print(f'CHANNEL NAMES ({patient}/{data_type}):')
        print('-'*100)
        ch_names_all = data.epochs[0].ch_names
        print(data.epochs[0].ch_names)
        print(len(data.epochs[0].ch_names))
        del data
        
        args.channel_name = [curr_labels]
        data = get_data(args)
        
        print(f'CHANNEL NAMES ({patient}/{data_type}):')
        print('-'*100)
        ch_names_picked = data.epochs[0].ch_names
        print(data.epochs[0].ch_names)
        print(len(data.epochs[0].ch_names))
        
        print('-'*100)
        print('Not picked')
        print(list(set(ch_names_all)-set(ch_names_picked)))
        del data

