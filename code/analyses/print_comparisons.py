import argparse
from utils.data_manip import prepare_metadata, extend_metadata
from utils import comparisons
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--comparison', type=str, default=None)
args = parser.parse_args()
print(args)


# GET METADATA BY READING THE LOGS FROM THE FOLLOWING PATIENT:
patient = 'patient_479_11'
print(f'Preparing metadata for patient {patient}')
metadata = prepare_metadata(patient)
metadata = extend_metadata(metadata)
#print(metadata.columns)

# COMPARISON
comparisons = comparisons.comparison_list()
if args.comparison is not None:
    comparison_names = [args.comparison]
else:
    comparison_names = comparisons.keys()

for comp_name in comparison_names:
    print('-'*100)
    print(comp_name)
    for block, block_name in zip([1, 4], ['Visual', 'Auditory']): # one visual, one auditory block
        print('-'*100)
        print(f'Block: {block_name.upper()}')
        for (cond_name, query) in zip(comparisons[comp_name]['condition_names'], comparisons[comp_name]['queries']):
            level = comparisons[comp_name]['level']
            print(cond_name, level, ':',  query)

            if level == 'sentence_onset':
                    tmin_, tmax_ = (-1.2, 3.5)
                    metadata_level = metadata.loc[((metadata['block'].isin([1, 3, 5])) &
                                                  (metadata['word_position'] == 1)) |
                                                  ((metadata['block'].isin([2, 4, 6])) &
                                                  (metadata['word_position'] == 1) &
                                                  (metadata['phone_position'] == 1))]
            elif level == 'sentence_offset':
                tmin_, tmax_ = (-3.5, 1.5)
                metadata_level = metadata.loc[(metadata['is_last_word'] == 1)]
            elif level == 'sentence_end':
                tmin_, tmax_ = (-3.5, 1.5)
                metadata_level = metadata.loc[(metadata['word_string'] == '.') | (metadata['phone_string'] == 'END_OF_WAV')]
            elif level == 'word':
                tmin_, tmax_ = (-1.5, 2.5)
                metadata_level = metadata.loc[((metadata['word_onset'] == 1) &
                                              (metadata['block'].isin([2, 4, 6]))) |
                                              ((metadata['block'].isin([1, 3, 5])) &
                                              (metadata['word_position'] > 0))]
            elif level == 'phone':
                tmin_, tmax_ = (-0.3, 1.2)
                metadata_level = metadata.loc[(metadata['block'].isin([2, 4, 6])) &
                                              (metadata['phone_position'] > 0)]

            # PRINT
            pick_columns = ['sentence_string', 'word_string', 'phone_string']
            query += f' & block=={block}'
            df_query = metadata_level.query(query, engine='python')[pick_columns].sort_values('sentence_string')
            print(df_query.to_string())
            print(f'Number of stimuli ({cond_name}): {len(df_query.index)}')
            print(f'tmin, tmax = {tmin_}, {tmax_}')
