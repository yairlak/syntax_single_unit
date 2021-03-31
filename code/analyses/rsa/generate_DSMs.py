import argparse, os, sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
from functions import load_settings_params, read_logs_and_features
from scipy.spatial.distance import pdist, squareform
import numpy as np


parser = argparse.ArgumentParser(description='Generate') 
# QUERY
parser.add_argument('--dimension', default='word_string', help='')
parser.add_argument('--comparison-name', default='word_string', help='')
parser.add_argument('--pick-classes', default=[], type=str, nargs='*', help='')
parser.add_argument('--path2DSMs', default='../../Paradigm/RSA/DSMs/', type=str)
# PARSE
args = parser.parse_args()

########
# INIT #
########
# GET METADATA BY READING THE LOGS FROM THE FOLLOWING PATIENT:
patient = 'patient_479_11'
print('Loading settings, params and preferences...')
settings = load_settings_params.Settings(patient)
params = load_settings_params.Params(patient)
preferences = load_settings_params.Preferences()

print('Metadata: Loading features and comparisons from Excel files...')
features = read_logs_and_features.load_features(settings)

print('Logs: Reading experiment log files from experiment...')
log_all_blocks = {}
for block in range(1, 7):
    log = read_logs_and_features.read_log(block, settings)
    log_all_blocks[block] = log

print('Loading POS tags for all words in the lexicon')
word2pos = read_logs_and_features.load_POS_tags(settings)

print('Preparing meta-data')
metadata = read_logs_and_features.prepare_metadata(log_all_blocks, features, word2pos, settings, params, preferences)


#################
# GENERATE DSM  #
#################
classes = list(set(metadata[args.dimension]))
num_classes = len(classes)
if args.dimension == 'word_string':
    mat_unigrams = np.zeros([num_classes, num_classes])
    mat_bigrams = np.zeros([num_classes, num_classes])
    mat_trigrams = np.zeros([num_classes, num_classes])
    for i_class in range(num_classes):
        for j_class in range(i_class + 1, num_classes):
            c_i, c_j = classes[i_class], classes[j_class]
            # UNIGRAMS
            letters_c_i, letters_c_j = set(list(c_i)), set(list(c_j))
            num_shared_unigrams = len(letters_c_i & letters_c_j)
            #print(c_i, c_j, num_shared_unigrams, len(c_i))
            # BIGARMS
            bigrams_c_i = [c_i[k]+c_i[k+1] for k in range(len(list(c_i))-1)]
            bigrams_c_j = [c_j[k]+c_j[k+1] for k in range(len(list(c_j))-1)]
            num_shared_bigrams = len(set(bigrams_c_i) & set(bigrams_c_j))
            #print(bigrams_c_i, bigrams_c_j, num_shared_bigrams)
            # TRIGARMS
            trigrams_c_i = [c_i[k]+c_i[k+1]+c_i[k+2] for k in range(len(list(c_i))-2)]
            trigrams_c_j = [c_j[k]+c_j[k+1]+c_j[k+2] for k in range(len(list(c_j))-2)]
            num_shared_trigrams = len(set(trigrams_c_i) & set(trigrams_c_j))
            #print(trigrams_c_i, trigrams_c_j, num_shared_trigrams)
            #
            # Dissimilarity = 1 - num_shared_features/mean_length_words
            mat_unigrams[i_class, j_class] = 1 - num_shared_unigrams/np.mean([len(c_i), len(c_j)])
            mat_unigrams[j_class, i_class] = 1 - num_shared_unigrams/np.mean([len(c_i), len(c_j)])
            mat_bigrams[i_class, j_class] = 1 - num_shared_bigrams/np.mean([len(c_i), len(c_j)])
            mat_bigrams[j_class, i_class] = 1 - num_shared_bigrams/np.mean([len(c_i), len(c_j)])
            mat_trigrams[i_class, j_class] = 1 - num_shared_trigrams/np.mean([len(c_i), len(c_j)])
            mat_trigrams[j_class, i_class] = 1 - num_shared_trigrams/np.mean([len(c_i), len(c_j)])
        
####################
# SAVE DSM TO FILE #
####################
if args.dimension == 'word_string':
    for feature in ['unigrams', 'bigrams', 'trigrams']:
        fname_DSM = os.path.join(args.path2DSMs, 'DSM_'+args.dimension+'_'+feature+'.txt')
        np.savetxt(fname_DSM, eval('mat_'+feature), delimiter=',')
        print('DSM saved to: ', fname_DSM)
    # ADD CLASS NAMES IN FIRST ROW OF TEXT FILE
    fname_classes = os.path.join(args.path2DSMs, 'CLASSES_'+args.dimension + '.txt')
    with open(fname_classes, 'w') as f:
            f.write(','.join(classes)+'\n')
    print('Class names in same order as DSMs saved to: ', fname_classes)
