import os
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
import pickle

path2wav = '../../Paradigm/Stimuli/Audio'
path2save = '../../Paradigm/Stimuli/Audio/mfcc'
if not os.path.exists(path2save):
    os.mkdir(path2save)

nfft = 2048
for i_trial in range(1, 153):
    (rate,sig) = wav.read(os.path.join(path2wav, f"{i_trial}.wav"))
    if i_trial == 1: print(f"Sampling Rate: {rate}")
    print(f"Generating mfcc features for trial {i_trial}")
    mfcc_feat = mfcc(sig,rate, nfft=nfft)
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig,rate, nfft=nfft)
    for feat_mat, name in zip([mfcc_feat, fbank_feat], ['mfcc', 'logfbank']):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(np.transpose(feat_mat))
        plt.savefig(os.path.join(path2save, f"{name}_{i_trial}.png"))
        plt.close(fig)
        np.savetxt(os.path.join(path2save, f"{name}_{i_trial}.png"), feat_mat)
        print(f"Figures and arrays saved to: {path2save}/{name}_{i_trial}.*")

