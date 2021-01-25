import argparse, os, sys, pickle
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
from functions.utils import dict2filename
from functions.data_manip import load_neural_data
from functions.read_logs_and_features import extend_metadata
from functions.features import get_features
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import mne
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train an encoding model on neural data')
# DATA
parser.add_argument('--patient', action='append', default=[], help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], action='append', default=[], help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='word', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'gaussian-kernel-25', 'high-gamma'], action='append', default=[], help='')
parser.add_argument('--probe-name', default=[], nargs='*', action='append', type=str, help='Probe name to plot (will ignore args.channel-name/num), e.g., LSTG')
parser.add_argument('--channel-name', default=[], nargs='*', action='append', type=str, help='Pick specific channels names')
parser.add_argument('--channe-num', default=[], nargs='*', action='append', type=int, help='channel number (if empty list [] then all channels of patient are analyzed)')
parser.add_argument('--responsive-channels-only', action='store_true', default=False, help='Include only responsive channels in the decoding model. See aud and vis files in Epochs folder of each patient')
# QUERY
parser.add_argument('--query', default=[], help='For example, to limit to first phone in auditory blocks "and first_phone == 1"')
parser.add_argument('--feature-list', default=[], nargs='*', help='Comparison name from Code/Main/functions/comparisons.py')
parser.add_argument('--block-type', choices=['auditory', 'visual', 'both'], default='both', help='Block type will be added to the query in the comparison')
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
assert len(args.patient)==len(args.data_type)==len(args.filter)==len(args.probe_name)
# FNAME 
#list_args2fname = ['patient', 'data_type', 'filter', 'level', 'block_type', 'model_type', 'ch_name', 'feature_list', 'query']
list_args2fname = ['patient', 'data_type', 'filter', 'level', 'block_type', 'model_type', 'ch_name', 'query']

np.random.seed(1)

#############
# LOAD DATA #
#############
epochs_list = load_neural_data(args)
#for epochs in epochs_list: # ADD MORE FEATURE COLUMNS TO METADATA
#    df = extend_metadata(epochs.metadata)
print(epochs_list[0])
metadata = epochs_list[0].metadata
times = epochs_list[0].times
# DEBUG
#times = times[:10] # DEBUG!
print(metadata['word_string'])
#print(metadata['word_position'])
X, feature_values, feature_info, feature_groups = get_features(metadata, args.feature_list) # GET DESIGN MATRIX
print('Design matrix dimensions:', X.shape)
num_samples, num_features = X.shape
#print('Feature names\n', feature_names)
print('Features\n')
[print(k, feature_info[k]) for k in feature_info.keys()]

###############
# STANDARDIZE #
###############
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

for epochs in epochs_list:
    for ch_name in epochs.ch_names:
        # ADD CH_NAME TO FNAME
        args.ch_name = ch_name
        args2fname = args.__dict__.copy()
        fname = dict2filename(args2fname, '_', list_args2fname, '', True)
        
        # GET NEURAL DATA
        pick_ch = mne.pick_channels(epochs.ch_names, [ch_name])
        y = np.squeeze(epochs.get_data()[:, pick_ch, :]) # num_trials X num_timepoints
        print('Target variable dimenstions:', y.shape)
        num_timepoints = y.shape[1]
        
        ########################
        # TRAIN AND TEST MODEL #
        ########################
        n_folds = 5 # CROSS-VALIDATION
        k_fold = KFold(n_folds)
        models, scores = [], []
        for k, (train, test) in enumerate(k_fold.split(X, y)):
            print(f'Train/test model: fold {k+1}/{n_folds}')
            scores_curr_split, models_curr_split = [], []
            for i_t, _ in enumerate(tqdm(times)):
                if args.model_type == 'ridge':
                    model = linear_model.Ridge()
                elif args.model_type == 'lasso':
                    model = linear_model.Lasso(alpha=0.01)
                # GRIDSEARCH FOR OPTIMAL REGULARIZER
                alphas = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
                search = GridSearchCV(model, param_grid={'alpha':alphas}, cv=5)
                search.fit(X[train, :], y[train, i_t])
                best_alpha = search.best_estimator_.alpha
                if args.model_type == 'ridge':
                    model = linear_model.Ridge(alpha=best_alpha)
                elif args.model_type == 'lasso':
                    model = linear_model.Lasso(alpha=best_alpha)
                #print('best alpha: ', best_alpha)
                # TRAIN
                model.fit(X[train, :], y[train, i_t])
                models_curr_split.append(model)
                # TEST
                scores_curr_split.append(model.score(X[test, :], y[test, i_t]))
            scores.append(np.asarray(scores_curr_split))
            models.append(models_curr_split)
        scores=np.asarray(scores)
        print(scores.shape)

        # PLOT BETAS
        betas = []
        for models_curr_split in models: 
            betas.append(np.asarray([model.coef_ for model in models_curr_split])) 
        betas = np.asarray(betas)
        betas_mean = np.mean(betas, axis=0).transpose() # num_features X num_timepoints
        betas_std = np.std(betas, axis=0).transpose()/np.sqrt(n_folds) # num_features X num_timepoints
        #betas_mean = np.abs(betas_mean) # TAKE ABS BETA
        
        ############
        # PLOTTING #
        ############
        
        fig, ax = plt.subplots(figsize=(20,10))
        for IX_feature, (betas_of_curr_feature, std_curr_feature, feature_value) in enumerate(zip(betas_mean, betas_std, feature_values)):
            #print(feature_value, feature_info)
            if feature_value in list(feature_info.keys()):
                color = feature_info[feature_value]['color']
                ls = feature_info[feature_value]['ls']
                lw = feature_info[feature_value]['lw']
            else:
                for k in feature_info.keys():
                    if IX_feature >= feature_info[k]['IXs'][0] and IX_feature < feature_info[k]['IXs'][1]:
                        if feature_info[k]['color']:
                            color = feature_info[k]['color']
                        else:
                            color = None
                        if ('ls' in feature_info[k].keys()) and feature_info[k]['ls']:
                            ls = feature_info[k]['ls']
                        else:
                            ls = '-'
                        if ('lw' in feature_info[k].keys()) and feature_info[k]['lw']:
                            lw = feature_info[k]['lw']
                        else:
                            lw = 3
            ax.plot(times*1e3, betas_of_curr_feature, color=color, ls=ls, lw=lw, label=feature_value)
            #print(feature_name, color)
            ax.fill_between(times*1e3, betas_of_curr_feature + std_curr_feature, betas_of_curr_feature - std_curr_feature , color=color, alpha=0.2)
        ax.legend(loc='center left', bbox_to_anchor=(1.12, 0.5), ncol=int(np.ceil(num_features/40)))
        ax.set_xlabel('Time (msec)', fontsize=20)
        ax.set_ylabel(r'Regression weight ($\beta$)', fontsize=20)
        ax.set_ylim((None, None)) 
        if args.block_type == 'visual':
            ax.axvline(x=0, ls='--', color='k')
            ax.axvline(x=500, ls='--', color='k')
        ax.axhline(ls='--', color='k')

        mean_scores = np.mean(scores, axis=0)
        std_scores = np.std(scores, axis=0)
        
        color = 'k'
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Coefficient of determination ($R^2$)', color=color, fontsize=20)  # we already handled the x-label with ax1
        ax2.plot(times*1e3, np.mean(scores, axis=0), color=color, lw=3)
        ax2.fill_between(times*1e3, mean_scores+std_scores, mean_scores-std_scores, color=color, alpha=0.2)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim((0, 1)) 
        plt.subplots_adjust(right=0.6)

        fname_fig = os.path.join(args.path2figures, fname + '.png')
        fig.savefig(fname_fig)
        plt.close(fig)
        print('Figure saved to: ', fname_fig)
        
        ###########################
        # PLOT PER FEATURE GROUPS #
        ###########################
        fig, ax = plt.subplots(figsize=(20,10))
       
        for group, feature_names in feature_groups.items():
            for feature_name in feature_names:
                if feature_name in feature_info.keys():
                    st, ed = feature_info[feature_name]['IXs']
                    if feature_info[feature_name]['color']:
                        color = feature_info[feature_name]['color']
                    else:
                        color = None
                    if ('ls' in feature_info[feature_name].keys()) and feature_info[feature_name]['ls']:
                        ls = feature_info[feature_name]['ls']
                    else:
                        ls = '-'
                    if ('lw' in feature_info[feature_name].keys()) and feature_info[feature_name]['lw']:
                        lw = feature_info[feature_name]['lw']
                    else:
                        lw = 3
                    mean_betas_across_features = []
                    mean_stds_across_features = []
                    for IX in range(st, ed):
                        mean_betas_across_features.append(np.abs(betas_mean[IX]))
                        mean_stds_across_features.append(betas_std[IX])
                    mean_betas_across_features = np.mean(np.asarray(mean_betas_across_features), axis=0)
                    mean_stds_across_features = np.mean(np.asarray(mean_stds_across_features), axis=0)
                    ax.plot(times*1e3, mean_betas_across_features, color=color, ls=ls, lw=lw, label=feature_name)
                    #print(feature_name, color)
                    ax.fill_between(times*1e3, mean_betas_across_features + mean_stds_across_features, mean_betas_across_features - mean_stds_across_features , color=color, alpha=0.2)

        ax.legend(loc='center left', bbox_to_anchor=(1.12, 0.5), ncol=int(np.ceil(num_features/40)))
        ax.set_xlabel('Time (msec)', fontsize=20)
        ax.set_ylabel(r'Regression weight ($\beta$)', fontsize=20)
        ax.set_ylim((None, None)) 
        if args.block_type == 'visual':
            ax.axvline(x=0, ls='--', color='k')
            ax.axvline(x=500, ls='--', color='k')
        ax.axhline(ls='--', color='k')

        mean_scores = np.mean(scores, axis=0)
        std_scores = np.std(scores, axis=0)
        
        color = 'k'
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Coefficient of determination ($R^2$)', color=color, fontsize=20)  # we already handled the x-label with ax1
        ax2.plot(times*1e3, np.mean(scores, axis=0), color=color, lw=3)
        ax2.fill_between(times*1e3, mean_scores+std_scores, mean_scores-std_scores, color=color, alpha=0.2)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim((0, 1)) 
        plt.subplots_adjust(right=0.6)

        fname_fig = os.path.join(args.path2figures, fname + '_groupped.png')
        fig.savefig(fname_fig)
        plt.close(fig)
        print('Figure saved to: ', fname_fig)
