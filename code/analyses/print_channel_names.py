import os
import argparse
from utils.data_manip import DataHandler

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser()
parser.add_argument('--patient', default='543', type=str)
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike'],
                    default='spike', help='electrode type')
args=parser.parse_args()

print(args)

fname = f'../../Data/UCLA/patient_{args.patient}/Raw/{args.data_type}/channel_numbers_to_names.txt'
if os.path.isfile(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
        ch_names = [ll.split()[-1] for ll in lines]
else:
    print(f'channel-names file was not found for patient {args.patient}, {args.data_type}')
    data = DataHandler('patient_' + args.patient, args.data_type, 'raw', None, None, None)
    data.load_raw_data()
    ch_names = data.raws[0].ch_names
    # SAVE TO FILE
    with open(fname, 'w') as f:
        for i_ch, ch_name in enumerate(ch_names):
            f.write(f'{i_ch+1} {ch_name}\n')

# PRINT TO SCREEN
[print(f'Channel {i_ch+1}: {ch_name}') for i_ch, ch_name in enumerate(ch_names)]
