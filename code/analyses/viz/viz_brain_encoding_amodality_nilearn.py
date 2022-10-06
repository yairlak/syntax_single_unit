#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
# import nibabel as nib
from utils.utils import read_MNI_coord_from_xls
# import ipyvolume as ipv
import mne
import matplotlib.pyplot as plt
import pandas as pd
# import pyvista as pv
from mne.stats import fdr_correction
from nilearn import plotting  

alpha = 0.05
thresh_r = 0.3 # For visualization only; max value of cbar
data_type = 'micro'
filt = 'raw'
fn_trf_results = f'../../../Output/encoding_models/evoked_encoding_results_decimate_50_smooth_50.json'
fn_trf_results = f'../../../Output/encoding_models/encoding_results_micro_raw_decimate_50_smooth_50_patients_479_11_479_25_482_499_502_505_510_513_515_530_538_539_540_541_543_544_549_551.json'

df = pd.read_json(fn_trf_results)

df = df.loc[df['data_type'] == data_type]
df = df.loc[df['filter'] == filt]
df = df.loc[df['Feature']=='full']

for block in ['auditory', 'visual']:
    pvals = np.asarray([np.asarray(a) for a in df[f'ps_full_{block}_word_trf'].values])
    pvals_cat = np.concatenate(pvals)
    reject_fdr, pvals_fdr = fdr_correction(pvals_cat,
                                           alpha=alpha,
                                           method='indep')
    df[f'pvals_full_{block}_fdr_spatiotemporal'] = pvals_fdr.reshape((pvals.shape[0], -1)).tolist()

def get_r_significant(row, block, alpha):
    rs = row[f'rs_full_{block}_word_trf']
    mask = np.logical_and(np.asarray(row[f'pvals_full_{block}_fdr_spatiotemporal']) <= alpha,
                          np.asarray(row[f'rs_full_{block}_word_trf']) > 0)
    if mask is None:
        return None
    rs_masked = np.asarray(rs)[mask]
    return rs_masked
    
for block in ['auditory', 'visual']:
    df[f'r_significant_full_{block}'] = \
        df.apply(lambda row: get_r_significant(row, block, alpha),
                 axis=1)

for block in ['auditory', 'visual']:
    df[f'r_mean_significant_full_{block}'] = \
        df.apply(lambda row: np.asarray(row[f'r_significant_full_{block}']).mean(),
                 axis=1)


def get_d_from_diag(row):
    visual = row['r_significant_full_visual']
    auditory = row['r_significant_full_auditory']
    if (visual is not None) and (auditory is not None):
        d = (visual.mean()-auditory.mean())/np.sqrt(2)
    else:
        d = None
    return d

df['d_from_diag'] = df.apply(lambda row: get_d_from_diag(row), axis=1)


def probename2ROI(path2mapping='../../../Data/probenames2fsaverage.tsv'):
    with open(path2mapping, 'r') as f:
        lines = f.readlines()
    atlas_regions = [line.split('\t')[0] for line in lines if line.split('\t')[1].strip('\n')!='-']
    probe_names = [line.split('\t')[1].strip('\n') for line in lines if line.split('\t')[1].strip('\n')!='-']
    p2r = {}
    for atlas_region, probe_name in zip(atlas_regions, probe_names):
        for probename in probe_name.split(','): # could be several, comma delimited
            assert probename not in p2r.keys()
            p2r[probename] = atlas_region
    return p2r

probename2ROI = probename2ROI()

def get_fsaverage_roi(row):
    if row['Probe_name'] in probename2ROI.keys():
        fsaverage_roi = probename2ROI[row['Probe_name']]
    else:
        fsaverage_roi = None
    return fsaverage_roi
    

df['ROI_fsaverage'] = df.apply(lambda row: get_fsaverage_roi(row), axis=1)
ROI_fsaverages = list(df['ROI_fsaverage'])
ROI_fsaverages = list(set([roi for roi in ROI_fsaverages if roi is not None]))

cmaps = {}
cmaps['auditory'] = 'Blues'
cmaps['visual'] = 'Reds'
def get_color(x, cmap='seismic'):
    return eval(f'plt.cm.{cmap}(x)')


show_labels = False
block = 'visual'
patients = [str(p)[:3] for p in df['Patient'].tolist()]
data_types = df['data_type'].tolist()
channel_names = df['Ch_name'].tolist()
rs = df[f'r_mean_significant_full_{block}'].values

dict_colors = {'micro':'r', 'macro':'b'}
path2data = os.path.join('..', '..', '..', 'Data', 'UCLA', 'MNI_coords')

labels, coords, colors = [], [], []
for patient, data_type, channel_name, r in zip(patients, data_types, channel_names, rs):
    if data_type == 'micro':
        channel_name = channel_name[4:]
        channel_name = ''.join([c for c in channel_name if not c.isdigit()])
    
    print(patient, data_type, channel_name)
    fn_coord = f'sub-{patient}_space-MNI_electrodes.xlsx'
    color = get_color(r/thresh_r, cmap=cmaps[block])[:3]
    
    df_coords = read_MNI_coord_from_xls(os.path.join(path2data, fn_coord),
                                        data_type, channel_name)
    
    curr_coords = df_coords[['MNI_x', 'MNI_y', 'MNI_z']].values
    curr_labels = df_coords['electrode'].values
    
    labels += list(curr_labels)
    coords += list(curr_coords)
    colors += [color] * len(curr_labels)
   

if not show_labels: labels=None
view = plotting.view_markers(coords, colors, marker_labels=labels,
                             marker_size=3) 

view.open_in_browser()
view.save_as_html(f'../../../Figures/viz_brain/encoding_{block}_{"_".join(patients)}.html')

