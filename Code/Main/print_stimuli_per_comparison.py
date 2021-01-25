import argparse, os, sys, glob, mne, pandas
# Set current working directory to that of script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname) 
sys.path.append('..')
from functions import comparisons

parser = argparse.ArgumentParser()
parser.add_argument('--comparison', required=True)
args = parser.parse_args()

# USE THE FOLLOWING CHANNEL AND PATIENT FOR WHAT FOLLOWS:
channel = 1
patient = 'patient_479_25'

# READ COMPARISON
comparisons = comparisons.comparison_list()
comparison = comparisons[int(args.comparison)]
str_comp = '### COMPARISON %s ###'%args.comparison
print('#'*len(str_comp))
print(str_comp)
print('#'*len(str_comp))
print(comparison['name'])
print()

# READ EPOCHS FILE
filename = patient + '_micro_*_ch_' + str(channel) + '-tfr.h5'
path2epochs = os.path.join('..', '..', 'Data', 'UCLA', patient, 'Epochs')
filenames = glob.glob(os.path.join(path2epochs, filename))
assert len(filenames)==1
path2epochs = filenames[0]
epochsTFR = mne.time_frequency.read_tfrs(path2epochs)
print(epochsTFR[0])
epochsTFR = epochsTFR[0]
print()

# QUERY EPOCH AND PRINT METADATA
front_cols = ['block', 'sentence_string', 'word_string', 'phone_string']
pandas.set_option('display.max_rows', None)

print('--- TRAIN ---')
for condition_name, query_train, color in zip(comparison['train_condition_names'], comparison['train_queries'], comparison['colors']):
    print('-'*150)
    print(condition_name, query_train, color)
    print('-'*150)
    metadata = epochsTFR[query_train].metadata
    metadata = metadata.sort_values(by=['event_time'])
    metadata = metadata[front_cols]
    print(metadata)
    print('Total number of stimuli: %i' % len(metadata))

if 'test_queries' in comparison.keys():
    print('--- TEST ---')
    for condition_name, query_test, color in zip(comparison['test_condition_names'], comparison['test_queries'], comparison['colors']):
        print('-'*150)
        print(condition_name, query_test, color)
        print('-'*150)
        metadata = epochsTFR[query_test].metadata
        metadata = metadata.sort_values(by=['event_time'])
        metadata = metadata[front_cols]
        print(metadata)
        print('Total number of stimuli: %i' % len(metadata))
        print()
print()
