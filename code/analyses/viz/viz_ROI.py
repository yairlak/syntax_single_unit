#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import nibabel as nib
import ipyvolume as ipv
import mne
import matplotlib.pyplot as plt
import pandas as pd
import pyvista as pv
from mne.stats import fdr_correction

# In[4]:
SUBJECTS_DIR = '/volatile/freesurfer/subjects' # your freesurfer directory
ROI_fsaverages = ['Brodmann.22-lh',
                  'Brodmann.45-lh']
label_colors = [np.array([255, 128, 128])/255,
                np.array([0, 0, 255])/255]

labels = mne.read_labels_from_annot(
    'fsaverage', 
    #parc='aparc.a2009s', 
    parc='PALS_B12_Brodmann',
    subjects_dir=SUBJECTS_DIR, 
    verbose=True
)


labels = [label for label in labels if label.name in ROI_fsaverages]

# In[8]:

subjects_dir = SUBJECTS_DIR
hemi='lh'
surface='pial'
 
fname = f'{subjects_dir}/fsaverage/surf/{hemi}.{surface}'
vertices, faces = nib.freesurfer.read_geometry(fname)
rr = vertices.copy()
tris = faces.astype(np.uint32)
faces = np.hstack([np.r_[3, i] for i in faces])

# Curvature
fname_curv = f'{subjects_dir}/fsaverage/surf/{hemi}.curv'
curv = nib.freesurfer.read_morph_data(fname_curv)

# Normalize curvature to make gray cortex
curv = (curv < 0).astype(float)
curv = (curv - 0.5) / 3 + 0.5
curv = curv[:, np.newaxis] * [1, 1, 1]

n_areas = len(ROI_fsaverages)
X, Y = np.meshgrid(np.linspace(0, 1, int(n_areas**.5)),
                   np.linspace(0, 1, int(n_areas**.5)),
                   sparse=False, indexing='xy')

colors = curv

for i_label, label in enumerate(labels):
    colors[label.vertices, :] = label_colors[i_label]

# In[14]:
fn = f"../../../Figures/viz_brain/ROI_{'_'.join(ROI_fsaverages)}"

surf = pv.PolyData(vertices, faces)
surf["colors"] = colors
p = pv.Plotter(off_screen=True)
p.set_background('w')

p.add_mesh(surf,
           show_edges=False,
           scalars="colors",
           cmap='RdBu_r',
           interpolate_before_map=False,
           clim=[-0.3, 0.3],
           rgb=True)

#ZY
p.view_zy()
p.camera.roll += 90
aspect = 'lateral'
zoom = {'medial':1.1, 'lateral':1.2}[aspect]
p.camera.zoom(zoom)
p.show(screenshot=fn+f'_lateral.png', auto_close=False)

  