import argparse, os, pickle
from utils.data_manip import DataHandler
import mne
from mne.decoding import (cross_val_multiscore, LinearModel, GeneralizingEstimator)
from utils import classification, load_settings_params, data_manip
from decoding.utils import get_comparisons, update_args
from decoding.data_manip import get_data
from utils.utils import dict2filename, update_queries, probename2picks
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.manifold import MDS
from sklearn.model_selection import ShuffleSplit
from sklearn.cluster import AgglomerativeClustering
from skorch import NeuralNetClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import torch
from decoding.data_manip import prepare_data_for_classification
#import models # custom module with neural-network models (LSTM/GRU/CNN)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(description='Generate plots for TIMIT experiment')
# DATA
<<<<<<< HEAD
<<<<<<< HEAD
parser.add_argument('--patient', action='append', default=['502'],
=======
# parser.add_argument('--patient', action='append', default=['502', '505', '515', '539', '540', '549'],
#                     help='Patient string')
parser.add_argument('--patient', action='append', default=['505'],
>>>>>>> 0402d6c821bb152bb80f3e58dd8137e5009891ff
=======
# parser.add_argument('--patient', action='append', default=['502', '505', '515', '539', '540', '549'],
#                     help='Patient string')
parser.add_argument('--patient', action='append', default=['505'],
>>>>>>> 0402d6c821bb152bb80f3e58dd8137e5009891ff
                    help='Patient string')

# parser.add_argument('--data-type', choices=['micro','macro', 'spike'],
#                     action='append', default=['micro', 'micro', 'micro', 'micro', 'micro', 'micro'], help='electrode type')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'],
                    action='append', default=['micro'], help='electrode type')

parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'],
                    default='word', help='')
<<<<<<< HEAD
<<<<<<< HEAD
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'gaussian-kernel-10'], action='append', default=[], help='')
parser.add_argument('--probe-name', default=[['RFSG']], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
=======
=======
>>>>>>> 0402d6c821bb152bb80f3e58dd8137e5009891ff
# parser.add_argument('--filter', action='append', default=['raw', 'raw', 'raw', 'raw', 'raw', 'raw'], help='')
parser.add_argument('--filter', action='append', default=['raw'], help='')
parser.add_argument('--smooth', default=None, help='')
parser.add_argument('--scale-epochs', action="store_true", default=False, help='')
# parser.add_argument('--probe-name', default=[['LFSG', 'RFSG'], ['LFGP'], ['LFSG'], ['LFSG'], ['RFSG'], ['LFSG']], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--probe-name', default=None, nargs='*', action='append', type=str,
                    help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--ROIs', default=None, nargs='*', type=str,
                    help='e.g., Brodmann.22-lh, overrides probe_name')
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str,
                    help='Pick specific channels names')
parser.add_argument('--channel-num', default=[], nargs='*', action='append', type=int,
                    help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False,
                    help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
<<<<<<< HEAD
# QUERY
>>>>>>> 0402d6c821bb152bb80f3e58dd8137e5009891ff
# QUERY
=======
# QUERY
# QUERY
>>>>>>> 0402d6c821bb152bb80f3e58dd8137e5009891ff
parser.add_argument('--comparison-name', default='word_string_all',
                    help='See Utils/comparisons.py')
parser.add_argument('--comparison-name-test', default=None,
                    help='See Utils/comparisons.py')
parser.add_argument('--block-train', choices=['auditory', 'visual'],
                    default='auditory',
                    help='Block type is added to the query in the comparison')
parser.add_argument('--block-test', choices=['auditory', 'visual'],
                    default=None,
                    help='Block type is added to the query in the comparison')
parser.add_argument('--fixed-constraint', default=None,
                    help='e.g., "and first_phone == 1"')
parser.add_argument('--fixed-constraint-test', default=None,
                    help='e.g., "and first_phone == 1"')
parser.add_argument('--min-trials', default=9, type=float,
                    help='Minimum number of trials from each class.')

# MODEL
parser.add_argument('--model-type', default='logistic', choices=['euclidean', 'logistic', 'lstm', 'cnn']) # 'svc' and 'ridge' are omited since they don't implemnent predict_proba (although there's a work around, using their decision function and map is to probs with eg softmax)
parser.add_argument('--cuda', default=False, action='store_true', help="If True then file will be overwritten")
# MISC
parser.add_argument('--vmin', default=None, type=float, help='')
parser.add_argument('--vmax', default=None, type=float, help='')
parser.add_argument('--times', nargs='*', default=[0.1], type=float, help='')
parser.add_argument('--time-window', default=0.6, type=float, help='')
parser.add_argument('--num-bins', default=5, type=int, help='')
parser.add_argument('--decimate', default=[], type=float, help='If not empty, (for speed) decimate data by the provided factor.')
parser.add_argument('--path2figures', default=[], help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--dont-overwrite', default=False, action='store_true', help="If True then file will be overwritten")
parser.add_argument('--path2output', default=[], help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
# PARSE
args = parser.parse_args()
args = update_args(args)
# args.patient = ['patient_' + p for p in  args.patient]
# print(mne.__version__)

########
# INIT #
########
USE_CUDA = args.cuda  # Set this to False if you don't want to use CUDA

# SET SEEDS
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

# Which args to have in fig filename
list_args2fname = ['patient', 'data_type', 'filter', 'level', 'comparison_name',
                   'block_train', 'time_window', 'num_bins', 'min_trials', 'query']
if args.block_test: list_args2fname += ['comparison_name_test', 'block_test']
if args.probe_name:
    list_args2fname.append('probe_name')
elif args.channel_name:
    list_args2fname.append('channel_name')
if args.responsive_channels_only: list_args2fname += ['responsive_channels_only']
print('args2fname', list_args2fname)

if not args.path2figures:
    args.path2figures = os.path.join('..', '..', 'Figures', 'RSA')
if not args.path2output:
    args.path2output = os.path.join('..', '..', 'Output', 'RSA')
print('args\n', args)


# GET COMPARISONS (CONTRASTS)
comparisons = get_comparisons(args.comparison_name, # List with two dicts for
                              args.comparison_name_test) # comparison train and test

print('\nARGUMENTS:')
pprint(args.__dict__, width=1)
if 'level' in comparisons[0].keys():
    args.level = comparisons[0]['level']
if len(comparisons[0]['queries'])>2:
    args.multi_class = True
else:
    args.multi_class = False

# LOAD DATA
print('\nLOADING DATA:')
args.tmin=None
args.tmax=None
data = get_data(args)

print('\nCONTRASTS:')
metadata = data.epochs[0].metadata
comparisons[0] = update_queries(comparisons[0], args.block_train, # TRAIN
                                args.fixed_constraint, metadata)
comparisons[1] = update_queries(comparisons[1], args.block_test, # TEST
                                args.fixed_constraint_test, metadata)
[pprint(comparison) for comparison in comparisons] 


if args.num_bins:
    bin_size = args.time_window / args.num_bins
for t in args.times:
    # PREPARE DATA
    # X_list = []
    ###############
    # BINNIZATION #
    ###############
    if args.num_bins:
        X = []
        for i_bin in range(args.num_bins):
            print('bin', i_bin)
            epochs_list = [epochs.copy().crop(t+i_bin*bin_size, t+(i_bin+1)*bin_size) for epochs in data.epochs]
            curr_X, y, labels = prepare_data_for_classification(epochs_list,
                                                                comparisons[0]['queries'],
                                                                args.model_type,
                                                                args.min_trials,
                                                                equalize_classes=False,
                                                                verbose=False)
            curr_X = np.mean(curr_X, axis=2, keepdims=True) # curr_X: (num_trials X num_channels X 1)
            X.append(curr_X)
        X = np.concatenate(X, axis=2) # X: num_trials x num_channels, num_bins
    else:
        epochs_list = [epochs.copy().crop(t, t+args.time_window) for epochs in data.epochs]
        X, y, labels = prepare_data_for_classification(epochs_list,
                                                       comparisons[0]['queries'],
                                                       args.model_type,
                                                       args.min_trials,
                                                       equalize_classes=False,
                                                       verbose=False)
        X = np.mean(X, axis=2, keepdims=True)
    labels = [label[0,1] for label in labels] # take only word string from tuples
    #     if not labels:
    #         labels = comparison['condition_names']
    #     X_list.append(X)
    #     #print('Shapes (X, y): ', X.shape, y.shape)
    #     [print(X.shape) for X in X_list]
    # X = np.concatenate(X_list, axis=1) # cat different patients/probes as new channel features (num_trials x num_channels, num_bins)
  
    ##################################
    # standardize and re-arange dims #
    ##################################
    new_X = []
    for ch in range(X.shape[1]):
        mat = X[:,ch,:]
        mat = stats.zscore(mat, axis=None)
        new_X.append(mat)
    X = np.dstack(new_X) # num_samples X num_bins X num_channel_features
    classes = list(set(y))
    print(classes, labels)
    print('Shapes (X, y): ', X.shape, y.shape)
    
    #if args.model_type in ['lstm', 'cnn']:
    num_samples, num_timepoints, num_channels = X.shape    
    if args.model_type in ['logistic', 'svc', 'ridge']:
        X = X.reshape(X.shape[0], -1)
        num_samples, num_features = X.shape    

    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y) # for imbalance classes, a 1D tensor
    class_weights = torch.FloatTensor(class_weights) # for imbalance classes, a 1D tensor
    num_classes=len(classes)

    X_means = np.empty([num_classes, num_channels * num_timepoints])
    print(X_means.shape)
    if args.model_type == 'euclidean':
        for i_c, c in enumerate(set(y)):
            IX_class = (y==c)
            X_class = X[IX_class, :, :]
            X_class_mean = np.mean(X_class, axis=0) # average across trials
            X_class_mean = np.reshape(X_class_mean, (1, -1)) # Reshape to a vector and place in matrix
            print(X_class_mean.shape)
            X_means[i_c, :] = X_class_mean
        DSM = squareform(pdist(X_means, metric=args.model_type))
        print(DSM.shape)
    else:
        ###############
        # GRID SEARCH #
        ###############
        if args.model_type == 'logistic':
            model = OneVsRestClassifier(LogisticRegression(solver='lbfgs', class_weight='balanced'))
            params = {'net__estimator__C': [0.01, 0.1, 1, 10, 100]}
        elif args.model_type == 'svc':
            model = OneVsRestClassifier(LinearSVC(class_weight='balanced'))
            params = {'net__estimator__C': [0.01, 0.1, 1, 10, 100]}
        elif args.model_type == 'ridge':
            model = OneVsRestClassifier(RidgeClassifier(class_weight='balanced'))
            params = {'net__estimator__alpha': [0.01, 0.1, 1, 10, 100]}
        elif args.model_type == 'lstm':
            model = models.RNNClassifier
            model = NeuralNetClassifier(model,
                                        optimizer=torch.optim.Adam,
                                        criterion=torch.nn.NLLLoss,
                                        criterion__weight=class_weights,
                                        device=('cuda' if args.cuda else 'cpu'),
                                        batch_size=10
                                        )
            params = {'net__module__input_dim': [num_channels],
                    'net__module__output_dim': [num_classes],
                    'net__module__num_units': [1],
                    'net__module__num_layers': [1],
                    'net__module__dropout': [0],
                    'net__lr': [0.01],
                    'net__max_epochs': [10],
                    }
        elif args.model_type == 'cnn':
            model = models.CNN()
            #print('Total params:', sum(p.numel() for p in model.parameters()))
            #print('Trainanle params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
            model = models.CNN
            model = NeuralNetClassifier(model,
                                        optimizer=torch.optim.Adam,
                                        criterion=torch.nn.NLLLoss,
                                        criterion__weight=class_weights,
                                        device=('cuda' if args.cuda else 'cpu'),
                                        batch_size=10
                                        )
            params = {'net__module__num_timepoints': [num_timepoints],
                    'net__module__in_channels': [num_channels],
                    'net__module__out_channels': [1, 2], # int(np.ceil(num_channels/2))],
                    'net__module__num_classes': [num_classes],
                    'net__module__kernel_size': [3, 5],
                    'net__module__kernel_size_maxp': [3, 5],
                    'net__module__stride': [1, 2],
                    'net__module__stride_maxp': [1, 2],
                    'net__module__dilation': [1],
                    'net__module__dilation_maxp': [1],
                    'net__module__dropout': [0, 0.25],
                    'net__lr': [0.1, 0.01],
                    'net__max_epochs': [10],
                     }
        
        print('Generating SKORCH pipeline')

        steps = [('net', model)]
        clf_grid = Pipeline(steps)
        print(model)
        #num_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #print(f'Total trainable params of model: {num_model_params}')

        #grid = RandomizedSearchCV(clf, params, n_iter=NUM_CV_STEPS, verbose=2, refit=False, scoring='accuracy', cv=5)
        grid = GridSearchCV(clf_grid, params, verbose=2, scoring='roc_auc_ovr', cv=5)
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        
        if args.model_type == 'cnn':
            X = np.swapaxes(X,1,2)
        # TRAIN AND EVAL
        # Compute confusion matrix for each cross-validation fold
        cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        y_pred = np.zeros((len(y), len(classes)))
        for i, (train, test) in enumerate(cv.split(X, y)):
            if i==0: # to find optimal hyperparams, run grid search on training data of the first split
                print('Grid search...')
                search = grid.fit(X[train], y[train])
                print(search)
                print(search.best_params_)

                #################
                # OPTIMAL MODEL #
                #################
                if args.model_type == 'logistic':
                    model = OneVsRestClassifier(LogisticRegression(solver='lbfgs', class_weight='balanced'))
                elif args.model_type == 'svc':
                    model = OneVsRestClassifier(LinearSVC(class_weight='balanced'))
                elif args.model_type == 'ridge':
                    model = OneVsRestClassifier(RidgeClassifier(class_weight='balanced'))
                elif args.model_type == 'lstm':
                    model = models.RNNClassifier(input_dim=num_channels,
                                                 output_dim=num_classes,
                                                 num_units=search.best_params_['net__module__num_units'],
                                                 num_layers=search.best_params_['net__module__num_layers'],
                                                 dropout=search.best_params_['net__module__dropout']
                                                 )
                    model = NeuralNetClassifier(model,
                                                     optimizer=torch.optim.Adam,
                                                     criterion=torch.nn.NLLLoss,
                                                     criterion__weight=class_weights,
                                                     device=('cuda' if args.cuda else 'cpu'),
                                                     batch_size=10,
                                                     lr=search.best_params_['net__lr'],
                                                     max_epochs=search.best_params_['net__max_epochs']
                                                     )
                elif args.model_type == 'cnn':
                    model = models.CNN(num_timepoints=num_timepoints,
                                       in_channels=num_channels,
                                       out_channels=int(np.ceil(num_channels/2)),
                                       num_classes=num_classes,
                                       kernel_size=search.best_params_['net__module__kernel_size'],
                                       kernel_size_maxp=search.best_params_['net__module__kernel_size_maxp'],
                                       stride=search.best_params_['net__module__stride'],
                                       stride_maxp=search.best_params_['net__module__stride_maxp'],
                                       dilation=search.best_params_['net__module__dilation'],
                                       dilation_maxp=search.best_params_['net__module__dilation_maxp'],
                                       dropout=search.best_params_['net__module__dropout']
                                       )
                    print(model)
                    [print(p.numel()) for p in model.parameters()]
                    print('Total params:', sum(p.numel() for p in model.parameters()))
                    print('Trainanle params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
                    model = NeuralNetClassifier(model,
                                                     optimizer=torch.optim.Adam,
                                                     criterion=torch.nn.NLLLoss,
                                                     criterion__weight=class_weights,
                                                     device=('cuda' if args.cuda else 'cpu'),
                                                     batch_size=10,
                                                     lr=search.best_params_['net__lr'],
                                                     max_epochs=search.best_params_['net__max_epochs']
                                                     )

                steps = [('net', model)]
                clf_optimal = Pipeline(steps)


            print('Split ' + str(i))
            # Fit
            X_train = X[train]
            y_train = y[train]
            #print(X_train, y_train)
            
            #print(i, X_train.shape, y_train.shape)
            clf_optimal.fit(X_train, y_train)
            # Probabilistic prediction (necessary for ROC-AUC scoring metric)
            X_test = X[test]
            y_pred[test] = clf_optimal.predict_proba(X_test)

        confusion = np.zeros((len(classes), len(classes)))
        for ii, train_class in enumerate(classes):
            for jj in range(ii, len(classes)):
                confusion[ii, jj] = roc_auc_score(y == train_class, y_pred[:, jj])
                confusion[jj, ii] = confusion[ii, jj]
        DSM = 1-confusion
        
    #chance = 0.5
    ####################
    # SAVE DSM TO FILE #
    ####################
    args2fname = args.__dict__.copy()
    if len(list(set(args2fname['data_type']))) == 1: args2fname['data_type'] = list(set(args2fname['data_type']))
    if len(list(set(args2fname['filter']))) == 1: args2fname['filter'] = list(set(args2fname['filter']))
    if args2fname['probe_name'] is not None:
        args2fname['probe_name'] = sorted(list(set([item for sublist in args2fname['probe_name'] for item in sublist]))) # !! lump together all probe names !! to reduce filename length
    if 'time' not in list_args2fname: list_args2fname.append('time')
    args2fname['time'] = t

    fname_conf = dict2filename(args2fname, '_', list_args2fname, 'pkl', True)
    os.makedirs(args.path2output, exist_ok=True)
    fname_conf = os.path.join(args.path2output, 'DSM_' + args.model_type + '_' + fname_conf)
    with open(fname_conf, 'wb') as f:
        pickle.dump([DSM, comparisons, args, classes, labels], f)

    ############################
    # CLUSTER CONFUSION MATRIX #
    ############################
    linkage = 'complete'
    # Setting distance_thershold = 0 ensures we compute the full tree
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=None, distance_threshold=0, affinity='precomputed')
    #distance_mat = np.arctanh(confusion) # mapping probabilities in [0, 1] to [0, inf] by using the inverse tanh 
    clustering.fit(DSM)
   

    #####################
    # PLOT MAT + DENDRO #
    #####################
    def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

        # Plot the corresponding dendrogram
        dendro = dendrogram(linkage_matrix, **kwargs)
        return dendro
    
    # PLOT DENDRO
    fig = plt.figure(figsize=(40, 30))
    ax_dendro = fig.add_axes([0.09,0.1,0.2,0.8])
    dendro = plot_dendrogram(clustering, ax=ax_dendro, orientation='left')
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])
    
    # PLOT SIMILARITY MATRIX
    index = dendro['leaves']
    if args.model_type == 'euclidean':
        S = np.exp(-DSM)
    else:
        S = 1 - DSM
    print(S[index,index].shape)
    print(S.shape)
    S = S[:, index] # reorder matrix based on hierarichal clustering
    print(S.shape)
    S = S[index, :] # reorder matrix based on hierarichal clustering
    print(S.shape)
    labels = np.asarray(labels)[index].tolist()
    ax = fig.add_axes([0.35,0.1,0.6,0.8])
    clim = [args.vmin, args.vmax]
    clim = [1-S.max(), S.max()]
    im = ax.matshow(S, cmap='RdBu_r', clim=clim, origin='lower')
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(labels, rotation=40, ha='left')
    ax.tick_params(labelsize=14)
    axcolor = fig.add_axes([0.96,0.1,0.02,0.8])
    plt.colorbar(im, cax=axcolor)
    
    args2fname = args.__dict__.copy() 
    if len(list(set(args2fname['data_type']))) == 1: args2fname['data_type'] = list(set(args2fname['data_type']))
    if len(list(set(args2fname['filter']))) == 1: args2fname['filter'] = list(set(args2fname['filter']))
    if args2fname['probe_name'] is not None:
        args2fname['probe_name'] = sorted(list(set([item for sublist in args2fname['probe_name'] for item in sublist]))) # !! lump together all probe names !! to reduce filename length
    if 'time' not in list_args2fname: list_args2fname.append('time')
    args2fname['time'] = t
    fname_fig = dict2filename(args2fname, '_', list_args2fname, 'png', True)
    fname_fig_rsa = os.path.join(args.path2figures, 'RSA_' + args.model_type.upper() + '_' + fname_fig)
    fig.savefig(fname_fig_rsa)
    print('Figures saved to: ' + fname_fig_rsa)
    plt.close('all')

    ##############
    # PLOT t-SNE #
    ##############
    fig, ax = plt.subplots(1)
    #mds = MDS(2, random_state=0, dissimilarity='precomputed')
    #summary = mds.fit_transform(1 - confusion)
    TSNE = TSNE(n_components=2, metric='precomputed', random_state=0)
    summary = TSNE.fit_transform(DSM)
    colors = dendro['color_list']
    for sel, (color, label) in enumerate(zip(colors, labels)):
        ax.text(summary[sel, 0], summary[sel, 1], label, color=color, fontsize=10)
    ax.axis('off')
    ax.set_xlim(summary[:, 0].min(), summary[:, 0].max())
    ax.set_ylim(summary[:, 1].min(), summary[:, 1].max())
    plt.tight_layout()
    fname_fig_mds = os.path.join(args.path2figures, 'tSNE_' +args.model_type + '_' + fname_fig)
    fig.savefig(fname_fig_mds)

