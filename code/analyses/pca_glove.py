import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.data_manip import prepare_metadata, extend_metadata, load_glove_model
import pandas as pd

parser = argparse.ArgumentParser(description='Convert TIMIT Mat files into MNE-Python')
parser.add_argument('-r', '--root-path', default='~/projects', help='Path to parent project folder')
parser.add_argument('-f', '--glove-fn', default='glove/vectors.txt', help='Path to parent project folder')
args = parser.parse_args()
print(args)

# GET METADATA BY READING THE LOGS FROM THE FOLLOWING PATIENT:
patient = 'patient_479_11'
metadata = prepare_metadata(patient)
metadata = extend_metadata(metadata)

print(list(metadata))




glove = load_glove_model(os.path.join(args.root_path, args.glove_fn))
X = []
for i_w, w in enumerate(metadata['word_string']):
    if list(metadata['word_length'])[i_w]>1:
        vec = glove[w.lower()]
    else:
        vec = np.zeros(25)
    X.append(vec)


print('mne-python version', mne.__version__)
path2stimuli = os.path.join(args.root_path, 'Data/Sounds')
path2data = os.path.join(args.root_path, 'Data')
patient = args.patient
path2epochs = os.path.join(args.root_path, 'Data', patient, 'Epochs')
path2figures = os.path.join(args.root_path, 'Figures', patient, args.method)
if not os.path.exists(path2figures):
    os.makedirs(path2figures)

print('Loading epochs object')
epochs_fname = os.path.join(path2epochs, patient + '-epo.fif')
epochs = mne.read_epochs(epochs_fname)
include_channel_names = [ch_name for ch_name in epochs.ch_names if args.roi[:-10] in ch_name]
picks = mne.pick_channels(ch_names=epochs.ch_names, include=include_channel_names)
#
tmins = np.arange(0,0.9, args.window_step)
tmaxs = [t+args.window_size for t in tmins]

for tmin, tmax in zip(tmins, tmaxs):

    cropped_epochs = epochs.copy().crop(tmin=tmin, tmax=tmax)
    if args.roi == 'all':
        pick_ch = range(len(epochs.ch_names))
    else:
        pick_ch = mne.pick_channels(epochs.ch_names, include_channel_names)
    X = cropped_epochs["word_string in ['END']"].load_data().get_data()[:, pick_ch, :] #drop_bad()
    y = cropped_epochs["word_string in ['END']"].metadata["sentence_string"].values
    X = np.average(X, 2)
    X_scaled = preprocessing.scale(X)

    if args.method == 'PCA':
        pca = decomposition.PCA(n_components=2)
        pca.fit(X_scaled)
        X_embedded = pca.transform(X_scaled)
        print(pca.explained_variance_ratio_)
        print(pca.singular_values_)
    elif args.method == 'ISOMAP':
        n_neighbors = 20
        X_embedded = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)


    #
    X_ave = []; y_ave = []
    labels = set(y)
    for label in labels:
        IX = (y == label)
        curr_ave = np.average(X_embedded[IX, :], 0)
        X_ave.append(curr_ave)
        y_ave.append(label)
    X_ave = np.asarray(X_ave)

    fig, ax = plt.subplots(figsize=(30, 30))

    for i, label in enumerate(y_ave):
        color = 'k'
        # if 'ly' in label.split(' '):
        #     color = 'r'
        ax.text(X_ave[i, 0], X_ave[i, 1], label, fontsize=10, color=color)

    x_min = np.min(X_ave[:, 0])
    x_max = np.max(X_ave[:, 0])
    y_min = np.min(X_ave[:, 1])
    y_max = np.max(X_ave[:, 1])
    plt.axis('off')
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))


    filename = args.patient + '_' + args.method + '_' + args.roi + '_tmin_tmax_' + str(int(tmin*1000)) + '_' + str(int(tmax*1000)) + 'ms.png'
    plt.savefig(os.path.join(path2figures, filename))
    print('fig saved to' + os.path.join(path2figures, filename))
    plt.close(fig)
    #
