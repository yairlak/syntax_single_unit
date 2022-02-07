import argparse, os, pickle
from utils.utils import dict2filename
#from utils.data_manip import load_neural_data
#from functions.features import get_features
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import mne
import matplotlib.pyplot as plt
from utils.data_manip import DataHandler

parser = argparse.ArgumentParser(description='Train an encoding model on neural data')
# DATA
parser.add_argument('--patient', action='append', default=['505'], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], action='append', default=['spike'], help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='word', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'gaussian-kernel-25', 'high-gamma'], action='append', default=['raw'], help='')
parser.add_argument('--probe-name', default=[], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channel-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
# QUERY
parser.add_argument('--query', default=[], help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--feature-list', default=['letter_by_position'], nargs='*', help='Comparison name from Code/Main/functions/comparisons.py')
parser.add_argument('--block-type', choices=['auditory', 'visual'], default='visual', help='Block type will be added to the query in the comparison')
parser.add_argument('--label-from-metadata', default=[], help='Field name in metadata that will be used to generate labels for the different classes. If empty, condition_names in comparison will be used')
parser.add_argument('--pick-classes', default=[], type=str, nargs='*', help='Limit the classes to this list')
# MODEL
parser.add_argument('--model-type', default='ridge', choices=['ridge', 'lasso', 'standard']) 
# MISC
parser.add_argument('--tmin', default=None, type=float, help='')
parser.add_argument('--tmax', default=None, type=float, help='')
parser.add_argument('--min-trials', default=15, type=float, help='Minimum number of trials from each class.')
parser.add_argument('--decimate', default=[], type=float, help='If not empty, (for speed) decimate data by the provided factor.')
parser.add_argument('--path2figures', default=os.path.join('..', '..', 'Figures', 'encoding_models'), help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--path2output', default=os.path.join('..', '..', 'Output', 'encoding_models'), help="Channels to analyze and merge into a single epochs object (e.g. -c 1 -c 2). If empty then all channels found in the ChannelsCSC folder")
parser.add_argument('--dont-overwrite', default=False, action='store_true', help="If True then file will be overwritten")


#############
# USER ARGS #
#############
args = parser.parse_args()
args.patient = ['patient_' + p for p in  args.patient]
#if not args.probe_name:
#    args.probe_name = ['All']
print('args\n', args)
assert len(args.patient)==len(args.data_type)==len(args.filter)#==len(args.probe_name)
# FNAME 
list_args2fname = ['patient', 'data_type', 'filter', 'level', 'block_type', 'model_type', 'ch_name', 'feature_list', 'query']

np.random.seed(1)


#############
# LOAD DATA #
#############
data = DataHandler(args.patient, args.data_type, args.filter,
                   args.probe_name, args.channel_name, args.channel_num,
                   args.feature_list)
data.load_raw_data(args.decimate)

# GET WORD-LEVEL DATA
data.epoch_data(level='word',
                query=args.query_train,
                smooth=args.smooth,
                scale_epochs=False,  # must be same as word level
                verbose=True)

# TAKE FEATURES FROM TIME ZERO (i.e., FROM WORD ONSET)
X = data.epochs[0].copy().pick_types(misc=True).get_data()
times = data.epochs[0].times
IXs = np.where(times==0)
X = X[:, :, IXs[0][0]] # Take feature values at t=0.
X = np.expand_dims(X, axis=2) # For compatibility below, add singelton
 
# GET NEURAL ACTIVITY (y)
y = data.epochs[0].copy().pick_types(seeg=True, eeg=True).get_data()
n_epochs, n_channels, n_times = y.shape
print(n_epochs, n_channels, n_times)



#############
# LOAD DATA #
#############
# epochs_list = load_neural_data(args)
# print(epochs_list)
# metadata = epochs_list[0].metadata
# times = epochs_list[0].times
# print(metadata['word_string'])
# #print(metadata['word_position'])
# X, feature_names, feature_groups = get_features(metadata, args.feature_list) # GET DESIGN MATRIX
# print('Design matrix dimensions:', X.shape)
# num_samples, num_features = X.shape
# #print('Feature names\n', feature_names)
# print('Features\n')
# [print(k, feature_groups[k]) for k in feature_groups.keys()]

###############
# STANDARDIZE #
###############
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

for epochs in data.epochs:
    for ch_name in epochs.ch_names:
        # ADD CH_NAME TO FNAME
        args.ch_name = ch_name
        args2fname = args.__dict__.copy()
        fname = dict2filename(args2fname, '_', list_args2fname, '', True)
        
        # GET NEURAL DATA
        pick_ch = mne.pick_channels(epochs.ch_names, [ch_name])
        #print(pick_ch)
        y = np.mean(epochs.get_data()[:, pick_ch, :], axis=-1)
        #print('Target variable dimenstions:', y.shape)
        
        ########################
        # TRAIN AND TEST MODEL #
        ########################
        n_folds = 10 # CROSS-VALIDATION
        k_fold = KFold(n_folds)
        models, scores = [], []
        for k, (train, test) in enumerate(k_fold.split(X, y)):
            if args.model_type == 'ridge':
                model = linear_model.Ridge()
            elif args.model_type == 'lasso':
                model = linear_model.Lasso()
            # TRAIN
            model.fit(X[train, :], y[train])
            models.append(model)
            # TEST
            scores.append(model.score(X[test, :], y[test]))
            
        scores=np.asarray(scores)
        #print(scores.shape)


        # MEAN BETAS ACROSS SPLITS
        betas = []
        for model in models: 
            betas.append(model.coef_) 
        betas = np.asarray(betas)
        #print(betas.shape)
        betas_mean = np.mean(betas, axis=0) # mean across splits
        betas_std = np.std(betas, axis=-1)/np.sqrt(n_folds) # num_features X num_timepoints
        #betas_mean = np.abs(betas_mean) # TAKE ABS BETA
        #print(betas_mean.shape)
        #print(betas_mean)
        beta_matrix = np.reshape(betas_mean, (3, -1), order='C') # 3 since inner/middle/outer letter
        #print(beta_matrix)

        ############
        # PLOTTING #
        ############
        word_strings = np.asarray(epochs.metadata['word_string'])
        all_letters = []
        [all_letters.extend(set(w)) for w in word_strings]
        all_letters = sorted(list(set(all_letters)-set(['.', '?']))) # REMOVE ? and . (!!)
        num_letters = len(all_letters)

        # PLOT BETAS
        fig, ax = plt.subplots(figsize=(10,10))
        for rw in range(beta_matrix.shape[0]):
            for col in range(beta_matrix.shape[1]):
                fontsize = 20 * abs(beta_matrix[rw, col])/np.max(np.abs(beta_matrix))
                color = 'r' if beta_matrix[rw, col] > 0 else 'b'
                #print(rw, col, all_letters[col], fontsize, color)
                ax.text(rw+1, col, all_letters[col], fontsize=fontsize, color=color)

        ax.set_xlabel('Position', fontsize=20)
        ax.set_ylabel('Letter identity', fontsize=20)
        plt.tick_params(axis='y', which='both', left=False, top=False, labelleft=False)
        ax.set_title(r'$R^2 = $ %1.2f +- %1.2f'%(np.mean(scores), np.std(scores)), fontsize=20)
        #ax.axes.get_yaxis().set_visible(False)
        ax.set_xticks(range(5))
        ax.set_xticklabels(['', 'First', 'Middle', 'Last', ''])
        ax.set_xlim((0, 4))
        ax.set_ylim((0, 25))
        #plt.subplots_adjust(right=0.6)
        
        fname_fig = os.path.join(args.path2figures, 'letter_by_position_' + fname + '.png')
        fig.savefig(fname_fig)
        plt.close(fig)
        print('Figure saved to: ', fname_fig)
        
