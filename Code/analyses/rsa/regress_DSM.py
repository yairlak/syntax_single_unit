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
from scipy.spatial.distance import squareform
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
parser.add_argument('--model-type', default='euclidean', choices=['euclidean', 'logistic', 'lstm', 'cnn']) # 'svc' and 'ridge' are omited since they don't implemnent predict_proba (although there's a work around, using their decision function and map is to probs with eg softmax)
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
parser.add_argument('--path2DSMs', default='../../Paradigm/RSA/DSMs', type=str, help='')
parser.add_argument('--dimension', default='word_string', type=str, help='')
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
list_args2fname = ['patient', 'data_type', 'filter', 'level', 'comparison_name', 'block_type', 'time_window', 'num_bins', 'min_trials']
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

#############
# LOAD DSMs #
#############
DSMs_features = []
if args.dimension == 'word_string':
    dimensions = ['word_string_' + s for s in ['unigrams', 'bigrams', 'trigrams']]
for d in dimensions:
    fn_DSM = 'DSM_' + d + '.txt'
    fn_DSM = os.path.join(args.path2DSMs, fn_DSM)
    with open(fn_DSM, newline='') as csvfile:
        curr_DSM = np.asarray(list(csv.reader(csvfile)))
    #curr_DSM = np.genfromtxt(fn_DSM, delimiter=',', dtype=None)
    DSMs_features.append(curr_DSM)

fn_classes = 'CLASSES_' + args.dimension + '.txt'
fn_classes = os.path.join(args.path2DSMs, fn_classes)
with open(fn_classes, 'r') as f:
    class_names_features = f.readlines()
    class_names_features = class_names_features[0].strip('\n').split(',')

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

def prepare_data_for_regression(Dy, Dxs, classes_y, classes_x):
    num_classes = Dy.shape[0]
    #assert len(pick_classes) == num_classes
    
    print(Dy.shape)
    np.fill_diagonal(Dy, 0) # manually change the values on the diagonal to zero
    y = squareform(Dy) # squareform is also the inverse of itself, in this case back to vector
    
    IXs_x2y = []
    for c in classes_y:
        IX = classes_x.index(c)
        IXs_x2y.append(IX)
    print(classes_x, classes_y, IXs_x2y)
    
    # TAKE ONLY COLUMNS AND ROWS FROM Dx THAT ARE RELEVANT FOR Dy
    Dxs = [Dx[IXs_x2y, :] for Dx in Dxs]
    Dxs = [Dx[:, IXs_x2y] for Dx in Dxs]
    Dxs = [squareform(Dx.astype(float)) for Dx in Dxs] # FLATTEN TO A VECTOR
    X = np.vstack(Dxs).transpose() # num_samples X num_features (num_samples = num_elements in upper triangular)
    
    return X, y
for t in args.times:

    ###################
    # LOAD NEURAL DSM #
    ###################
    args2fname = args.__dict__.copy() 
    if len(list(set(args2fname['data_type']))) == 1: args2fname['data_type'] = list(set(args2fname['data_type']))
    if len(list(set(args2fname['filter']))) == 1: args2fname['filter'] = list(set(args2fname['filter']))
    args2fname['probe_name'] = sorted(list(set([item for sublist in args2fname['probe_name'] for item in sublist]))) # !! lump together all probe names !! to reduce filename length
    if 'time' not in list_args2fname: list_args2fname.append('time')
    args2fname['time'] = t
    
    fname_DSM_data = dict2filename(args2fname, '_', list_args2fname, 'pkl', True)
    fname_DSM_data = os.path.join(args.path2output, 'DSM_' + args.model_type + '_' + fname_DSM_data)
    print('Loading: ', fname_DSM_data)
    with open(fname_DSM_data, 'rb') as f:
        DSM_data, comparison, args_conf, classes_data, class_names_data = pickle.load(f)
    if args.pick_classes: # pick only subclass
        IX_classes = np.asarray([i for i, l in enumerate(labels) if l in args.pick_classes])
        DSM_data = DSM_data[np.ix_(IX_classes, IX_classes)]
        labels = [l for i, l in enumerate(labels) if i in IX_classes]
        # classes = classes[IX_classes] CHECK IF MAKES SENSE
    
    ###########
    # REGRESS # 
    ###########
    X, y = prepare_data_for_regression(DSM_data, DSMs_features, class_names_data, class_names_features)
    print('Dimensions of Ax=B:', X.shape, y.shape)
    
    # SCALRE AND REGRESS
    print("Computing regularization path using the coordinate descent lasso...")
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    #model = linear_model.LassoCV(positive=True)
    #model = linear_model.LinearRegression()
    model = linear_model.RidgeCV()
   
    k_fold = KFold(5)
    results={}; models=[]
    for k, (train, test) in enumerate(k_fold.split(X, y)):
        model.fit(X[train], y[train])
        models.append(model)
        results[k] = {}
        #results[k]['alpha'] = model.alpha_
        results[k]['scores'] = model.score(X[test], y[test])

    print(results) 
    [print(m.coef_) for m in models]
    ################
    # SAVE RESULTS #
    ################
    fname_regress = dict2filename(args2fname, '_', list_args2fname, 'pkl', True)
    fname_regress = os.path.join(args.path2output, 'RegCoef_' + args.model_type + '_' + fname_regress)
    with open(fname_regress, 'wb') as f:
        pickle.dump([models, results, args_conf, class_names_features], f)
    print('Saved to: ', fname_regress)
