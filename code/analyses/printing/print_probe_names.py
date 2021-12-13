import sys, os, argparse
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('../')
from pprint import pprint
from utils import data_manip, utils
from utils.utils import get_all_patient_numbers

parser = argparse.ArgumentParser()
parser.add_argument('--patient', action='append', default=[])
parser.add_argument('--spikes', action='store_true', default=False)
args=parser.parse_args()

if not args.patient:
    args.patient = get_all_patient_numbers(path2data='../../../Data/UCLA')

print(args)
micro_from_all_patients, macro_from_all_patients = [], []
for patient in args.patient:
    probe_names_micro, channel_names_micro = utils.get_probe_names(patient, 'micro')
    probe_names_macro, channel_names_macro = utils.get_probe_names(patient, 'macro')

    print('%s' % patient)
    micro_from_all_patients.extend(probe_names_micro)
    macro_from_all_patients.extend(probe_names_macro)
    print('micro:', probe_names_micro)
    print('macro:', probe_names_macro)
    print('-'*100)

if args.spikes:
    x = {}
    probes = data_manip.get_probes2channels(patients)
    for probe in probes['probe_names'].keys():
        if 'patients' in probes['probe_names'][probe].keys():
            x[probe] = ' '.join(probes['probe_names'][probe]['patients'])
    dict_names = {k: v for k, v in sorted(x.items(), key=lambda item: len(item[1].split()), reverse=True)}
    [print('%s (%i):%s'%(k,len(v.split()),v)) for (k,v) in dict_names.items()]

print('All micro probes:')
print(' '.join(list(set(micro_from_all_patients))))
print('-'*100)
print('All macro probes:')
print(' '.join(list(set(macro_from_all_patients))))
