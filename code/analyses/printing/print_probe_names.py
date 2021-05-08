import sys, os, argparse
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('../')
from pprint import pprint
from utils import data_manip, utils

parser = argparse.ArgumentParser()
parser.add_argument('--patient', action='append', default=[])
parser.add_argument('--spikes', action='store_true', default=False)
parser.add_argument('--plot-ch-names', action='store_true', default=False)
args=parser.parse_args()

if not args.patient:
    args.patient = ['479_11', '479_25', '482', '487', '493', '502', '504', '505', '510', '513', '515']

print(args)
names_from_all_patients = []
for patient in args.patient:
    probe_names_micro, channel_names_micro = utils.get_probe_names(patient, 'micro')
    probe_names_macro, channel_names_macro = utils.get_probe_names(patient, 'macro')

    print('%s' % patient)
    if not args.plot_ch_names:
        if (probe_names_micro is not None) & (probe_names_macro is not None):
            if set(probe_names_micro) != set(probe_names_macro):
                print('!!! - Micro and macro electrode names are not the same - !!!')
                print('micro:', probe_names_micro)
                print('macro:', probe_names_macro)
                names_from_all_patients.extend(probe_names_micro)
            else:
                print(probe_names_micro)
                names_from_all_patients.extend(probe_names_micro)
        else:
            print('micro:', probe_names_micro)
            print('macro:', probe_names_macro)
        print('-'*100)
    else:
        print('micro:')
        [print(i+1, c) for i,c in enumerate(channel_names_micro)]
        print('macro:')
        [print(i+1, c) for i,c in enumerate(channel_names_macro)]


if args.spikes:
    x = {}
    probes = data_manip.get_probes2channels(patients)
    for probe in probes['probe_names'].keys():
        if 'patients' in probes['probe_names'][probe].keys():
            x[probe] = ' '.join(probes['probe_names'][probe]['patients'])
    dict_names = {k: v for k, v in sorted(x.items(), key=lambda item: len(item[1].split()), reverse=True)}
    [print('%s (%i):%s'%(k,len(v.split()),v)) for (k,v) in dict_names.items()]

print('-'*100)

print(' '.join(list(set(names_from_all_patients))))
print("', '".join(list(set(names_from_all_patients))))
