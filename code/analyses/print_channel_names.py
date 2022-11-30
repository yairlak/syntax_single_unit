import os
import argparse
from utils.data_manip import DataHandler
from utils.utils import get_all_patient_numbers

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser()
parser.add_argument('--patient', default=None, type=str)
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike'],
                    default='spike', help='electrode type')
parser.add_argument('--from-raw',
                    action='store_true',
                    default=False)
args=parser.parse_args()

print(args)

if not args.patient:
    patients = get_all_patient_numbers()
else:
    patients = [args.patient]


for patient in patients:
    print('-'*100)
    print(f'Channels for patient {patient}')
    print('-'*100)
    fname = f'../../Data/UCLA/patient_{patient}/Raw/{args.data_type}/channel_numbers_to_names.txt'
    try: 
        if os.path.isfile(fname) and not args.from_raw:
            with open(fname, 'r') as f:
                lines = f.readlines()
                ch_names = [ll.split()[-1] for ll in lines]
        else:
            print(f'channel-names file was not found for patient {patient}, {args.data_type}')
            data = DataHandler('patient_' + patient, args.data_type, 'raw', None, None, None)
            data.load_raw_data()
            ch_names = data.raws[0].ch_names
            # SAVE TO FILE
            if not args.from_raw:
                with open(fname, 'w') as f:
                    for i_ch, ch_name in enumerate(ch_names):
                        f.write(f'{i_ch+1} {ch_name}\n')

        # PRINT TO SCREEN
        [print(f'Channel {i_ch+1}: {ch_name}') for i_ch, ch_name in enumerate(ch_names)]
    except:
        print('No channels found')
