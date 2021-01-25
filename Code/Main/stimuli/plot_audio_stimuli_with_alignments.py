import os
import textgrids
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import pylab

path2stimuli = '../../Paradigm/Stimuli/Audio/normalized/resampled_16k/'

for sentence_num in range(1, 153):
    f_wav = '%i.wav' % sentence_num
    f_lab = '%i.lab' % sentence_num
    f_textgrid = '%i.TextGrid' % sentence_num
    fn_output_fig = '%i.png' % sentence_num

    grid = textgrids.TextGrid(os.path.join(path2stimuli, f_textgrid))
    phones = grid['phones']
    words = grid['words']
    print(phones, words)

    sample_rate, samples = wavfile.read(os.path.join(path2stimuli, f_wav))


    fig, axs = plt.subplots(2, 1, figsize=(20,10))
    times_sec = np.asarray(range(len(samples)))/sample_rate
    axs[0].plot(times_sec, samples/max(abs(samples)))
    pylab.specgram(samples, NFFT=80, Fs=16000, noverlap=40)


    phones_str = []
    phones_times = []
    for phone in phones:
        axs[1].axvline(phone.xmin, ymax=8000, color='k', ls='--')
        # plt.text(phone.xmin, 7500, phone.text, verticalalignment='center')
        phones_str.append(phone.text)
        phones_times.append(phone.xmin)

    for word in words:
        if not word.text in ['sil', 'sp']:
            axs[0].axvline(word.xmin, ymax=8000, color='r', ls='--')
            axs[0].text(word.xmin, 1.1, word.text, verticalalignment='center', fontsize=16)
            axs[1].axvline(word.xmin, ymax=8000, color='r', ls='--')
            axs[1].text(word.xmin, 6500, word.text, verticalalignment='center', fontsize=16)

    plt.setp(axs[0], ylabel='Signal', xlim=[0, max(times_sec)], ylim=[-1, 1])
    axs[0].set_xlabel('Time [sec]', fontsize=14)
    axs[0].set_ylabel('Acoustic Waveform', fontsize=14)
    plt.setp(axs[1], xlim=[0, max(times_sec)], xticks=phones_times, xticklabels=phones_str)
    plt.setp(axs[1].get_xticklabels(), fontsize=14)
    axs[1].set_ylabel('Frequency [Hz]', fontsize=14)

    plt.savefig(os.path.join(path2stimuli, fn_output_fig))
    print('Figure as saved to: %s' % os.path.join(path2stimuli, fn_output_fig))
