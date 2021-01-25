from functions import load_settings_params, read_logs_and_features
from functions.features import get_features

# INIT
patient = 'patient_479_11'
settings = load_settings_params.Settings('patient_479_11')
params = load_settings_params.Params('patient_479_11')
features = read_logs_and_features.load_features(settings)
# LOAD LOGS
log_all_blocks = {}
for block in range(1, 7):
    log = read_logs_and_features.read_log(block, settings)
    log_all_blocks[block] = log

# METADATA
metadata = read_logs_and_features.prepare_metadata(log_all_blocks, settings, params)
metadata = read_logs_and_features.extend_metadata(metadata)
print(list(metadata))

# WORD FEATURES
f1, f2 = read_logs_and_features.load_word_features(settings)
#print(f2)

# DESIGN MATRIX FOR ENCODING MODELS
# pos, letters, letter_by_position, bigrams, morphemes, first_letter, last_letter, word_length, sentence_length, word_position, word_zipf
feature_list = ['letters', 'semantic_features', 'word_length', 'morph_complex', 'is_first_word', 'is_last_word', 'word_position', 'tense', 'pos_simple', 'word_zipf', 'grammatical_number', 'embedding', 'wh_subj_obj', 'dec_quest', 'phone_string']
design_matrix, feature_values, feature_groups = get_features(metadata, feature_list)
print(design_matrix)
print(feature_values)
[print('-'*150, '\n',k, feature_groups[k]) for k in feature_groups.keys()]

