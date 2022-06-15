#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:56:54 2022

@author: yl254115
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

path2output = '../../../Output/decoding/'
fname = '_patient_479_11_patient_479_25_patient_482_patient_499_patient_505_patient_515_patient_530_patient_538_patient_541_patient_543_patient_544_micro_raw_phone_LHGa_LSTG_LSTGa_LSTGp_RASTG_RPSTG_RSTG.pkl'

results_manner = pickle.load(open(os.path.join(path2output,
                                               'manner' + fname), 'rb'))
results_place = pickle.load(open(os.path.join(path2output,
                                              'place' + fname), 'rb'))



fig, axs = plt.subplots(2, 2, figsize=(25, 20))
fig_diag, axs_diag = plt.subplots(1, 2, figsize=(25, 10))

d = {'manner':'place', 'place':'manner'}
color_dict = {'manner':'g', 'place':'r'}

vmin, vmax = 0.3, 0.7

for i_feature_type, feature_type in enumerate(['manner', 'place']):    
    
    times = eval(f'results_{feature_type}[2]')
    
    mats_diag, mats_off_diag = [], []

    for i in range(3):
        for j in range(3):
            if i==j:
                mats_diag.append(eval(f'results_{feature_type}[1][i, j]'))
            else:
                mats_off_diag.append(eval(f'results_{feature_type}[1][i, j]'))
                
            
    mats_diag = np.mean(np.stack(mats_diag, axis=-1), axis=-1)
    mats_off_diag = np.mean(np.stack(mats_off_diag, axis=-1), axis=-1)
    
    # PLOT mean GATs
    im = axs[i_feature_type, 0].imshow(mats_diag,
                                  origin='lower',
                                  cmap='RdBu_r',
                                  vmin=vmin,
                                  vmax=vmax)
    # axs[i_feature_type, 0].set_title(f'Decoding {feature_type} features', fontsize=30)
    
    axs[i_feature_type, 1].imshow(mats_off_diag,
                                  origin='lower',
                                  cmap='RdBu_r',
                                  vmin=vmin,
                                  vmax=vmax)
    axs[i_feature_type, 1].set_title(f'Generalization across {d[feature_type]} features', fontsize=30)
    
    
    t_0 = np.where(times == 0)[0][0]
    t_max = times.size-1
    for j in range(2):
        axs[i_feature_type, j].set_xticks([t_0, t_max])
        axs[i_feature_type, j].set_xticklabels(['0', times[t_max]])
        axs[i_feature_type, j].set_yticks([t_0, t_max])
        axs[i_feature_type, j].set_yticklabels(['0', times[t_max]])
        axs[i_feature_type, j].tick_params(labelsize=20)
        axs[i_feature_type, j].axhline(t_0, color='k', ls='--', lw=2)
        axs[i_feature_type, j].axvline(t_0, color='k', ls='--', lw=2)
    
    axs[0, 0].set_ylabel('Training Time (s)', fontsize=40)
    axs[1, 0].set_ylabel('Training Time (s)', fontsize=40)
    axs[1, 0].set_xlabel('Test Time (s)', fontsize=40)
    axs[1, 1].set_xlabel('Test Time (s)', fontsize=40)
    
    for i in range(2):
        for j in range(2):
            axs[i, j].axhline(0., color='k')
            axs[i, j].axvline(0., color='k')
    
    # PLOT diags
    axs_diag[0].plot(times, np.diag(mats_diag), color=color_dict[feature_type],
                     label=feature_type.capitalize(), lw=3)
    axs_diag[1].plot(times, np.diag(mats_off_diag), color=color_dict[feature_type],
                     label=feature_type.capitalize(), lw=3)
    
    axs_diag[0].set_xlabel('Time (s)', fontsize=30)
    axs_diag[1].set_xlabel('Time (s)', fontsize=30)
    axs_diag[0].set_ylabel('AUC', fontsize=30)
    axs_diag[1].set_ylabel('AUC', fontsize=30)
    axs_diag[0].axvline(0., color='k', ls='--')
    axs_diag[1].axvline(0., color='k', ls='--')
    axs_diag[0].axhline(0.5, color='k', ls='--')
    axs_diag[1].axhline(0.5, color='k', ls='--')
    axs_diag[0].tick_params(labelsize=20)
    axs_diag[1].tick_params(labelsize=20)
    axs_diag[0].set_ylim((0.45, 0.75))
    axs_diag[1].set_ylim((0.45, 0.75))
    axs_diag[0].legend(fontsize=20)
    axs_diag[1].legend(fontsize=20)
    
    #ax.set_title(f'{args.comparison_name} {args.block_type} {args.comparison_name_test} {args.block_type_test}')
plt.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.9, 0.35, 0.05, 0.3])
cbar = fig.colorbar(im, cax=cbar_ax)
# cbar.set_ticks([vmin, 0.5, vmax])
cbar.set_ticks([])
cbar_ax.tick_params(labelsize=20)


fn_fig = '../../../Figures/Decoding/manner_vs_place_from_3by3.png'
fig.savefig(fn_fig)
plt.close(fig)

fn_fig = '../../../Figures/Decoding/manner_vs_place_diags.png'
fig_diag.savefig(fn_fig)
plt.close(fig_diag)