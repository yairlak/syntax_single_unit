import argparse, os, sys, pickle
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
import csv
import mne
from functions import classification, comparisons, load_settings_params
from functions.utils import dict2filename, update_queries, probename2picks, pick_responsive_channels
from functions.data_manip import load_epochs_data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

parser = argparse.ArgumentParser(description='Generate plots for TIMIT experiment')
# DATA
parser.add_argument('--patient', action='append', default=[], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], action='append', default=[], help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='word', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'high-gamma'], action='append', default=[], help='')
parser.add_argument('--probe-name', default=[], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channe-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
# QUERY
parser.add_argument('--comparison-name', default='all_words', help='Comparison name from Code/Main/functions/comparisons.py')
parser.add_argument('--comparison-name-test', default=[], help='Comparison name from Code/Main/functions/comparisons.py')
parser.add_argument('--block-type', choices=['auditory', 'visual'], default='auditory', help='Block type will be added to the query in the comparison')
parser.add_argument('--block-type-test', choices=['auditory', 'visual', []], default=[], help='Block type will be added to the query in the comparison')
parser.add_argument('--fixed-constraint', default=[], help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--fixed-constraint-test', default=[], help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--classifier', default='logistic', choices=['svc', 'logistic', 'ridge'], help='Specify a classifier type to be used')
parser.add_argument('--label-from-metadata', default=[], help='Field name in metadata that will be used to generate labels for the different classes. If empty, condition_names in comparison will be used')
# FEATURES
parser.add_argument('--path2features', default=None, type=str, help='')
parser.add_argument('--pick-features', default=[], type=str, nargs='*', help='')
parser.add_argument('--pick-classes', default=[], type=str, nargs='*', help='')
# MISC
parser.add_argument('--vmin', default=None, type=float, help='')
parser.add_argument('--vmax', default=None, type=float, help='')
parser.add_argument('--times', nargs='*', default=[0.1], type=float, help='')
parser.add_argument('--num-bins', default=1, type=int, help='')
parser.add_argument('--time-window', default=0.1, type=float, help='')
parser.add_argument('--min-trials', default=10, type=float, help='Minimum number of trials from each class.')
parser.add_argument('--decimate', default=[], type=float, help='If not empty, (for speed) decimate data by the provided factor.')
parser.add_argument('--cat-k-timepoints', type=int, default=1, help='How many time points to concatenate before classification')
parser.add_argument('--path2figures', default=[], help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--path2output', default=[], help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--dont-overwrite', default=False, action='store_true', help="If True then file will be overwritten")
# PARSE
args = parser.parse_args()
print(mne.__version__)

# Which args to have in fig filename
list_args2fname = ['patient', 'data_type', 'filter', 'level', 'comparison_name', 'block_type']
if args.block_type_test: list_args2fname += ['comparison_name_test', 'block_type_test']
if args.probe_name: list_args2fname.append('probe_name')
if args.responsive_channels_only: list_args2fname += ['responsive_channels_only']
print(list_args2fname)
#args.patient = ['patient_' + p for p in  args.patient]

if not args.path2figures:
    args.path2figures = os.path.join('..', '..', 'Figures', 'Decoding')
if not args.path2output:
    args.path2output = os.path.join('..', '..', 'Output', 'RSA')
print(args)

#################
# LOAD FEATURES #
#################
class2features = {}
with open(args.path2features, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for i_row, row in enumerate(reader):
        if i_row == 0:
            class2features['class_type'] = row[0]
            class2features['feature_names'] = row[1:]
        elif i_row<34:
            class2features[row[0]] = np.asarray(row[1:]).astype(float)

# if chosen by user, remove unwanted classes
if args.pick_classes:
    pick_classes = [e for e in class2features.keys() if e not in ['class_type', 'feature_names']]
    for key in pick_classes:
        if key not in args.pick_classes:
            del class2features[key]
classes = [e for e in class2features.keys() if e not in ['class_type', 'feature_names']]
print('Class names: ', classes)

# if chosen by user, remove unwanted features
if args.pick_features:
    features = class2features['feature_names']
    IX_features = [i for i, f in enumerate(features) if f in args.pick_features]
    class2features['feature_names'] = [f for i, f in enumerate(class2features['feature_names']) if i in IX_features]
    for key in classes:
        class2features[key] = class2features[key][IX_features]

###########
# REGRESS #
###########

def prepare_data_for_regression(confusion, class2features, pick_classes):
    num_classes = confusion.shape[0]
    assert len(pick_classes) == num_classes
    

    # Similarity based on Shepard's method: (p_ij+p_ji)/(p_ii+p_jj)
    D = []
    for i in range(num_classes):
        for j in range (i+1, num_classes):
            p_ij = confusion[i, j]
            p_ji = confusion[j, i]
            p_ii = confusion[i, i]
            p_jj = confusion[j, j]
            S_ij = (p_ij + p_ji)/(p_ii + p_jj)
            D.append(-np.log(S_ij))
    D = np.asarray(D)
    D = D**2


    # Design matrix
    P_tilde = []
    for i in range(num_classes):
        class_i = pick_classes[i]
        p_i = class2features[class_i]
        for j in range(i+1, num_classes):
            class_j = labels[j]
            p_j = class2features[class_j]
            P_tilde.append((p_i-p_j)**2)
    P_tilde = np.asarray(P_tilde)         
    
    return P_tilde, D



for t in args.times:

    #########################
    # LOAD CONFUSION MATRIX #
    #########################
    args2fname = args.__dict__.copy() 
    if len(list(set(args2fname['data_type']))) == 1: args2fname['data_type'] = list(set(args2fname['data_type']))
    if len(list(set(args2fname['filter']))) == 1: args2fname['filter'] = list(set(args2fname['filter']))
    args2fname['probe_name'] = sorted(list(set([item for sublist in args2fname['probe_name'] for item in sublist]))) # !! lump together all probe names !! to reduce filename length
    if 'time' not in list_args2fname: list_args2fname.append('time')
    args2fname['time'] = t
    
    fname_conf = dict2filename(args2fname, '_', list_args2fname, 'pkl', True)
    fname_conf = os.path.join(args.path2output, 'CONF_LSTM_' + fname_conf)
    print('Loading: ', fname_conf)
    with open(fname_conf, 'rb') as f:
        confusion, comparison, args_conf, classes, labels = pickle.load(f)

    if args.pick_classes: # pick only subclass
        IX_classes = np.asarray([i for i, l in enumerate(labels) if l in args.pick_classes])
        confusion = confusion[np.ix_(IX_classes, IX_classes)]
        labels = [l for i, l in enumerate(labels) if i in IX_classes]
        # classes = classes[IX_classes] CHECK IF MAKES SENSE
    
    ###########
    # REGRESS # 
    ###########
    P_tilde, D = prepare_data_for_regression(confusion, class2features, labels)
    #scaler = StandardScaler().fit(P_tilde)
    #P_tilde = scaler.transform(P_tilde)
    print('Dimensions of Ax=B:', P_tilde.shape, D.shape)
    print("Computing regularization path using the coordinate descent lasso...")
    model = linear_model.LassoCV(positive=True)
    #model = linear_model.LinearRegression()
   
    k_fold = KFold(5)
    results={}; models=[]
    for k, (train, test) in enumerate(k_fold.split(P_tilde, D)):
        model.fit(P_tilde[train], D[train])
        models.append(model)
        results[k] = {}
        #results[k]['alpha'] = model.alpha_
        results[k]['scores'] = model.score(P_tilde[test], D[test])

 
    ################
    # SAVE RESULTS #
    ################
    fname_regress = dict2filename(args2fname, '_', list_args2fname, 'pkl', True)
    fname_regress = os.path.join(args.path2output, 'RegCoef_LSTM_' + fname_regress)
    with open(fname_regress, 'wb') as f:
        pickle.dump([models, results, class2features, confusion, labels, comparison, args_conf], f)
    print('Saved to: ', fname_regress)
