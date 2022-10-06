#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import nibabel as nib
# import ipyvolume as ipv
import mne
import matplotlib.pyplot as plt
import pandas as pd
# import pyvista as pv
from mne.stats import fdr_correction

# In[4]:
SUBJECTS_DIR = '/volatile/freesurfer/subjects' # your freesurfer directory

# In[5]:
data_type = 'micro'
filt = 'raw'
fn_trf_results = f'../../Output/encoding_models/evoked_encoding_results_decimate_50_smooth_50.json'
fn_trf_results = f'../../Output/encoding_models/encoding_results_micro_raw_decimate_50_smooth_50_patients_479_11_479_25_482_499_502_505_510_513_515_530_538_539_540_541_543_544_549_551.json'
df = pd.read_json(fn_trf_results)

def get_dr(r_full, r):
    if (not r_full is None) and (not r is None):
        if isinstance(r_full, str):
            r = [float(e) for e in r[1:-1].split()]
            r_full = [float(e) for e in r_full[1:-1].split()]
            dr = np.asarray(r_full) - np.asarray(r)
        elif isinstance(r_full, float):
            if np.isnan(r_full) or np.isnan(r):
                return np.nan
            else:
                dr = np.asarray(r_full) - np.asarray(r)
        else:
            dr = np.asarray(r_full) - np.asarray(r)
    else:
        dr=np.nan
    return dr
# df['dr_visual_total'] = df.apply (lambda row: get_dr(row['r_full_visual'],
#                                                row['r_visual']), axis=1)
# df['dr_auditory_total'] = df.apply (lambda row: get_dr(row['r_full_auditory'],
#                                                  row['r_auditory']), axis=1)

df['dr_visual_by_time'] = df.apply (lambda row: get_dr(row['r_full_visual_by_time'],
                                               row['r_visual_by_time']), axis=1)
df['dr_auditory_by_time'] = df.apply (lambda row: get_dr(row['r_full_auditory_by_time'],
                                                 row['r_auditory_by_time']), axis=1)

def find_max(dr_by_time):
    if isinstance(dr_by_time, float):
        if np.isnan(dr_by_time):
            return np.nan
        else:
            return dr_by_time.max()
    else:
        return dr_by_time.max()
                
df['dr_visual_max'] = df.apply (lambda row: find_max(row['dr_visual_by_time']), axis=1)
df['dr_auditory_max'] = df.apply (lambda row: find_max(row['dr_auditory_by_time']), axis=1)



# df['d_from_diag'] = df.apply(lambda row: (row['r_full_visual']-row['r_full_auditory'])/np.sqrt(2), axis=1)

# In[6]:
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


def get_color(x, cmap='RdBu_r'):
    return eval(f'plt.cm.{cmap}((np.clip(x,-1,1)+1)/2.)')


features = ['phonology', 'lexicon', 'syntax', 'semantics', 'positional']

for feature in features:
    fig_plt, axs = plt.subplots(1, 4, figsize=(20, 5))
    [ax.set_axis_off() for ax in axs]
    