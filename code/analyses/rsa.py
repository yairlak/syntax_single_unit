import argparse, os, sys, pickle
from utils.data_manip import DataHandler
import mne
from mne.decoding import (cross_val_multiscore, LinearModel, GeneralizingEstimator)
from utils import classification, comparisons, load_settings_params, data_manip
from utils.utils import dict2filename, update_queries, probename2picks, pick_responsive_channels
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
import models # custom module with neural-network models (LSTM/GRU/CNN)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(description='Generate plots for TIMIT experiment')
# DATA
<<<<<<< HEAD
parser.add_argument('--patient', action='append', default=['502'],
                    help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'],
                    action='append', default=['micro'], help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'],
                    default='word', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'gaussian-kernel-10'], action='append', default=[], help='')
parser.add_argument('--probe-name', default=[['RFSG']], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
=======
parser.add_argument('--patient', action='append', default=[], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], action='append', default=[], help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='word', help='')
parser.add_argument('--filter', action='append', default=[], help='raw/gaussian-kernel/high-gamma/etc')
parser.add_argument('--probe-name', default=[], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
>>>>>>> 52582a65026db42387227d69f974895004273a8a
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
# QUERY
parser.add_argument('--comparison-name', default='all_words', help='Comparison name from Code/Main/functions/comparisons.py')
parser.add_argument('--comparison-name-test', default=[], help='Comparison name from Code/Main/functions/comparisons.py')
parser.add_argument('--block-type', choices=['auditory', 'visual'], default='visual', help='Block type will be added to the query in the comparison')
parser.add_argument('--block-type-test', choices=['auditory', 'visual', []], default=[], help='Block type will be added to the query in the comparison')
parser.add_argument('--query-train', default='block in [1,3,5]', help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--fixed-constraint', default=[], help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--label-from-metadata', default=[], help='Field name in metadata that will be used to generate labels for the different classes. If empty, condition_names in comparison will be used')
parser.add_argument('--pick-classes', default=[], type=str, nargs='*', help='Limit the classes to this list')
# MODEL
parser.add_argument('--model-type', default='logistic', choices=['euclidean', 'logistic', 'lstm', 'cnn']) # 'svc' and 'ridge' are omited since they don't implemnent predict_proba (although there's a work around, using their decision function and map is to probs with eg softmax)
parser.add_argument('--cuda', default=False, action='store_true', help="If True then file will be overwritten")
# MISC
parser.add_argument('--vmin', default=None, type=float, help='')
parser.add_argument('--vmax', default=None, type=float, help='')
parser.add_argument('--times', nargs='*', default=[0.1], type=float, help='')
parser.add_argument('--time-window', default=0.5, type=float, help='')
parser.add_argument('--num-bins', default=[], type=int, help='')
parser.add_argument('--min-trials', default=6, type=float, help='Minimum number of trials from each class.')
parser.add_argument('--decimate', default=[], type=float, help='If not empty, (for speed) decimate data by the provided factor.')
parser.add_argument('--path2figures', default=[], help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--dont-overwrite', default=False, action='store_true', help="If True then file will be overwritten")
parser.add_argument('--path2output', default=[], help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
# PARSE
args = parser.parse_args()
args.patient = ['patient_' + p for p in  args.patient]
print(mne.__version__)

########
# INIT #
########
USE_CUDA = args.cuda  # Set this to False if you don't want to use CUDA

# SET SEEDS
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

# Which args to have in fig filename
list_args2fname = ['patient', 'data_type', 'filter', 'level', 'comparison_name', 'block_type', 'time_window', 'num_bins', 'min_trials', 'query']
if args.block_type_test: list_args2fname += ['comparison_name_test', 'block_type_test']
if args.probe_name:
    list_args2fname.append('probe_name')
elif args.channel_name:
    list_args2fname.append('channel_name')
if args.responsive_channels_only: list_args2fname += ['responsive_channels_only']
print('args2fname', list_args2fname)

if not args.path2figures:
    args.path2figures = os.path.join('..', '..', '..', 'Figures', 'RSA')
if not args.path2output:
    args.path2output = os.path.join('..', '..', '..', 'Output', 'RSA')
print('args\n', args)


#############
# LOAD DATA #
#############
data = DataHandler(args.patient, args.data_type, args.filter,
                   args.probe_name, args.channel_name, args.channel_num)
# Both neural and feature data into a single raw object
data.load_raw_data()
# GET SENTENCE-LEVEL DATA BEFORE SPLIT
data.epoch_data(level=args.level,
                query=args.query_train,
                scale_epochs=False,
                verbose=True)

print('Channel names:')
[print(e.ch_names) for e in data.epochs]

######################
# Queries TRAIN/TEST #
######################

# COMPARISON
comparisons = comparisons.comparison_list()
comparison = comparisons[args.comparison_name].copy()
comparison = update_queries(comparison, args.block_type,
                            args.fixed_constraint, data.epochs[0].metadata)
print('Comparison:')
print(comparison)

# Generalization to TEST SET
if args.comparison_name_test:
    comparison_test = comparisons[args.comparison_name_test].copy()
    if not args.block_type_test:
        raise('block-type-test is missing. Comparison-name-test was provided')
    comparison_test = update_queries(comparison_test, args.block_type_test,
                                     args.fixed_constraint_test)


def prepare_data_for_classifier(epochs, comparison, pick_classes,
                                field_for_labels=[]):
    '''
    '''
    X = []; y = []; labels = []; cnt = 0
    for q, query in enumerate(comparison['queries']):
        if 'heard' in query: # HACK! for word_string
            continue
        if 'END_OF_WAV' in query: # HACK! for phone_string
            continue
        if epochs[query].get_data().shape[0] < args.min_trials:
            #print(f'Less than {args.min_trials} trials matched query: {query}')
            continue
        if field_for_labels: # add each value of a feature as a label (e.g., for word_length - 2, 3, 4..)
            label = list(set(epochs[query].metadata[field_for_labels]))
            assert len(label) == 1
            label = label[0]
        else:
            label = comparison['condition_names'][q]
        if pick_classes and (label not in pick_classes):
            continue
        labels.append(label)
        curr_data = epochs[query].get_data()
        X.append(curr_data)
        num_trials = curr_data.shape[0]
        y.append(np.full(num_trials, cnt))
        cnt += 1

    X = np.concatenate(X, axis=0) # cat along the trial dimension
    y = np.concatenate(y, axis=0)

    return X, y, labels

if args.num_bins:
    bin_size = args.time_window / args.num_bins
for t in args.times:
    # PREPARE DATA
    X_list = []
    for epochs in data.epochs: # loop over epochs from different patients or probes
        ###############
        # BINNIZATION #
        ###############
        if args.num_bins:
            X = []
            for i_bin in range(args.num_bins):
                print('bin', i_bin)
                curr_epochs = epochs.copy().crop(t+i_bin*bin_size, t+(i_bin+1)*bin_size)
                curr_X, y, labels = prepare_data_for_classifier(curr_epochs, comparison, args.pick_classes, args.label_from_metadata)
                curr_X = np.mean(curr_X, axis=2, keepdims=True) # curr_X: (num_trials X num_channels X 1)
                X.append(curr_X)
            X = np.concatenate(X, axis=2) # X: num_trials x num_channels, num_bins
        else:
            X, y, labels = prepare_data_for_classifier(epochs.copy().crop(t, t+args.time_window), comparison, args.pick_classes, args.label_from_metadata)

        if not labels:
            labels = comparison['condition_names']
        X_list.append(X)
        #print('Shapes (X, y): ', X.shape, y.shape)
        [print(X.shape) for X in X_list]
    X = np.concatenate(X_list, axis=1) # cat different patients/probes as new channel features (num_trials x num_channels, num_bins)
  
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
    args2fname['probe_name'] = sorted(list(set([item for sublist in args2fname['probe_name'] for item in sublist]))) # !! lump together all probe names !! to reduce filename length
    if 'time' not in list_args2fname: list_args2fname.append('time')
    args2fname['time'] = t

    fname_conf = dict2filename(args2fname, '_', list_args2fname, 'pkl', True)
    fname_conf = os.path.join(args.path2output, 'DSM_' + args.model_type + '_' + fname_conf)
    with open(fname_conf, 'wb') as f:
        pickle.dump([DSM, comparison, args, classes, labels], f)

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

