import os
import scipy.io as sio
import numpy as np

path2features = '../../../../Paradigm/Stimuli/Audio/normalized/resampled_16k'

auditory_spectrum_all_trials = []
for trial in range(1, 153):
    fn = f'{trial}_acoustic_features.mat'
    data = sio.loadmat(os.path.join(path2features, fn))
    # dict keys in data are : ['paras', 'Fs', 'auditory_spectrum', 'signal_waveform']:
    auditory_spectrum_all_trials.append(np.transpose(data['auditory_spectrum']))
    sfreq_auditory_spectrum = 1/data['paras'][0][0] # first element of paras is the frame width in ms

num_frames = [X.shape[1] for X in auditory_spectrum_all_trials]
auditory_spectrum_all_trials = [X[:, :min(num_frames)] for X in auditory_spectrum_all_trials]
auditory_spectrum_all_trials = np.stack(auditory_spectrum_all_trials, axis = 0) # n_epochs X n_features (freqs) X n_timepoints
print(auditory_spectrum_all_trials.shape)

