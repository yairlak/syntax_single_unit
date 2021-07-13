import argparse
from utils.data_manip import DataHandler


parser = argparse.ArgumentParser()
parser.add_argument('--patient', default='505')
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike'],
                    default='spike', help='electrode type')
parser.add_argument('--filter', default='raw', help='raw/high-gamma')
args=parser.parse_args()

print(args)
data = DataHandler('patient_' + args.patient, args.data_type, args.filter,
                   None, None, None)
# Both neural and feature data into a single raw object
data.load_raw_data()
ch_names = data.raws[0].ch_names
[print(f'Channel {i_ch+1}: {ch_name}') for i_ch, ch_name in enumerate(ch_names)]
