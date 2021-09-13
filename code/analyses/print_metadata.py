import argparse
from utils.data_manip import prepare_metadata, extend_metadata
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--query', '-q', type=str, default=None)
parser.add_argument('--columns', '-c', type=str, nargs='*', default=None)
parser.add_argument('--print-all', '-a', default=False, action='store_true')
args = parser.parse_args()

# GET METADATA BY READING THE LOGS FROM THE FOLLOWING PATIENT:
patient = 'patient_479_11'
metadata = prepare_metadata(patient)
metadata = extend_metadata(metadata)

print(list(metadata))
#
for k in sorted(list(metadata)):
    if k == 'semantic_features': print(k, ' : not showing (too many values)'); continue
    try:
        k_values = list(set(metadata[k].values))
    except:
        print(f'Skipping: {k}')
        continue

    if any([v!=v for v in k_values]):
        print(k, len(k), 'contains nan values')
    else:
        if len(k_values) < 10:
            print(k, ':', k_values)
        else:
            print(k,  '(too many values, showing first 10):', k_values[:10])
print('-'*100)
print('num samples:', len(metadata[k]))
print('-'*100)
print('-'*100)


if args.query:
    metadata = metadata.query(args.query)
    if args.columns:
        metadata = metadata[args.columns]
    if args.print_all:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(metadata)
    else:
        print(metadata)
