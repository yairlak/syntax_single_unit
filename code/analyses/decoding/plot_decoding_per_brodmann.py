#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:58:17 2022

@author: yl254115
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as image

path2figure = '../../../Figures/Decoding/temp/'

smooth, decimate = 50, 50
comparisons = ['number_all', 'dec_quest_len2', 'embedding_vs_long', 'pos_simple']

def generate_decoding_subplot(comparison, brodmann):
    empty_fig = True
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for i_dtf, data_type_filter in enumerate(['micro_raw', 'micro_high-gamma']):
        for i_block, block in enumerate(['auditory', 'visual']):
            for i_hemi, hemi in enumerate(['lh', 'rh']):
                fn = f'Slider_{comparison}_{block}_{data_type_filter}_Brodmann.{brodmann}-{hemi}_{smooth}_{decimate}_True.png'
                row = i_block
                col = i_dtf * 2 + i_hemi
                
                try:
                    im = image.imread(os.path.join(path2figure, fn))
                    axs[row, col].imshow(im)
                    axs[row, col].axis('off')
                    axs[row, col].set_title(f'{data_type_filter} - {hemi}',
                                            fontsize=20)
                    empty_fig = False
                except:
                    print(f'Figure {fn} not found')
                    plt.close(fig)
    fig.text(0.025, 0.7, 'Auditory', fontsize=20)
    fig.text(0.025, 0.3, 'Visual', fontsize=20)
    if empty_fig:
        return None
    else:
        return fig

for comparison in comparisons:
    print(comparison)
    for brodmann in range(1, 48):
        fig = generate_decoding_subplot(comparison, brodmann)
        if fig is not None:
            fig.savefig(os.path.join(path2figure, f'decoding_{comparison}_brodmann_{brodmann}.png'))
            plt.close(fig)
