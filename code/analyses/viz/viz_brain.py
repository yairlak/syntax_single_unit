#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import nibabel as nib
import ipyvolume as ipv
import mne
import matplotlib.pyplot as plt
import pandas as pd


# In[4]:
SUBJECTS_DIR = '/volatile/freesurfer/subjects' # your freesurfer directory

# In[5]:
data_type = 'micro'
filt = 'raw'
fn_trf_results = f'../../../Output/encoding_models/trf_results_{data_type}_{filt}.csv'
df = pd.read_csv(fn_trf_results)

def get_dr(r_full, r):
    if isinstance(r_full, float):
        if np.isnan(r_full) or np.isnan(r):
            return np.nan
        else:
            dr = r_full - r
    else:
        dr = r_full - r
    return dr
df['dr_visual_total'] = df.apply (lambda row: get_dr(row['r_full_visual'],
                                               row['r_visual']), axis=1)
df['dr_auditory_total'] = df.apply (lambda row: get_dr(row['r_full_auditory'],
                                                 row['r_auditory']), axis=1)

df['d_from_diag'] = df.apply(lambda row: (row['r_full_visual']-row['r_full_auditory'])/np.sqrt(2), axis=1)

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

# In[7]:


labels = mne.read_labels_from_annot(
    'fsaverage', 
    #parc='aparc.a2009s', 
    parc='PALS_B12_Brodmann',
    subjects_dir=SUBJECTS_DIR, 
    verbose=True
)


labels = [label for label in labels if label.name in ROI_fsaverages]

# In[8]:


# def read_surface(subjects_dir, hemi='lh', surface='pial'):
subjects_dir = SUBJECTS_DIR
hemi='lh'
surface='pial'
# Read Anatomy
fname = f'{subjects_dir}/fsaverage/surf/{hemi}.{surface}'
rr, tris = nib.freesurfer.read_geometry(fname)
tris = tris.astype(np.uint32)

# Curvature
fname_curv = f'{subjects_dir}/fsaverage/surf/{hemi}.curv'
curv = nib.freesurfer.read_morph_data(fname_curv)

# Normalize curvature to make gray cortex
curv = (curv < 0).astype(float)
curv = (curv - 0.5) / 3 + 0.5
curv = curv[:, np.newaxis] * [1, 1, 1]


# In[9]:


n_areas = len(ROI_fsaverages)
X, Y = np.meshgrid(np.linspace(0, 1, int(n_areas**.5)),
                   np.linspace(0, 1, int(n_areas**.5)),
                   sparse=False, indexing='xy')

# for i in range(len(X)):
#     for j in range(len(X[0])):
#         plt.scatter(i, j, color=[X[i,j], 0, Y[i,j]], s=200)
  


# In[10]:


color = curv
# for label, x, y in zip(labels, X.ravel(), Y.ravel()):
#     color[label.vertices, :] = [x, 0, y]


def get_color(x, cmap='seismic'):
    return eval(f'plt.cm.{cmap}((np.clip(x,-1,1)+1)/2.)')

for label in labels:
    df_roi = df.loc[df['Feature'] == 'full']
    df_roi = df_roi.loc[df_roi['ROI_fsaverage'] == label.name]
    color[label.vertices, :] = [df_roi['r_full_visual'].mean(),
                                0,
                                df_roi['r_full_auditory'].mean()]
    d = -1 * df_roi['d_from_diag'].mean()
    color[label.vertices, :] = get_color(d)[:3]


# In[14]:


fig = ipv.figure(width=500, height=500)
brain = ipv.plot_trisurf(rr[:, 0], 
                         rr[:, 1], 
                         rr[:, 2],
                         triangles=tris,
                         color=color)

ipv.squarelim()
# fig.xlabel = ''
# fig.ylabel = ''
# fig.zlabel = ''
ipv.style.axes_off()
ipv.style.box_off()
ipv.show()
fn = f'../../../Figures/viz_brain/modality_specificity_{hemi}.html'
ipv.pylab.save(fn)
print(f'Saved to {fn}')

# In[ ]:




