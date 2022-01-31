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

# In[5]:
data_type = 'micro'
fn_trf_results = f'../../../Output/encoding_models/evoked_encoding_results_decimate_50_smooth_50.json'
df = pd.read_json(fn_trf_results)

df = df.loc[df['data_type'] == data_type]

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


subjects_dir = SUBJECTS_DIR
hemis=['lh', 'rh']
surface='pial'


def get_color(x, cmap='RdBu_r'):
    return eval(f'plt.cm.{cmap}((np.clip(x,-1,1)+1)/2.)')


fig_plt, axs = plt.subplots(1, 4, figsize=(20, 5))
[ax.set_axis_off() for ax in axs]

for i_hemi, hemi in enumerate(hemis):
    
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
    
    
    
    for label in labels:
        df_roi = df.loc[df['Feature'] == 'full']
        df_roi = df_roi.loc[df_roi['ROI_fsaverage'] == label.name]
        colors[label.vertices, :] = get_color(len(df_roi)/100)[:3]
    
    # In[14]:
    
    
    fn = f'../../../Figures/viz_brain/coverage_{hemi}_{data_type}'
    
    surf = pv.PolyData(vertices, faces)
    surf["colors"] = colors
    p = pv.Plotter(off_screen=True)
    p.set_background('w')
    
    
    p.add_mesh(surf,
               #color='grey',
               show_edges=False,
               scalars="colors",
               cmap='RdBu_r',
               interpolate_before_map=False,
               clim=[0, 1],
               rgb=True)
               #annotations = {chance_level:'chance'})
    
    # https://docs.pyvista.org/examples/02-plot/scalar-bars.html
        # p.export_html('test.html')
    
    #ZY
    p.view_zy()
    p.camera.roll += 90
    aspect = {'lh':'lateral', 'rh':'medial'}[hemi]
    zoom = {'medial':1.1, 'lateral':1.2}[aspect]
    p.camera.zoom(zoom)
    p.show(screenshot=fn+f'_{aspect}.png', auto_close=False)
    axs[i_hemi*2].imshow(p.image)
    
    # YZ
    p.view_yz()
    
    
    aspect = {'lh':'medial', 'rh':'lateral'}[hemi]
    #zoom = {'medial':1.65, 'lateral':1.8}[aspect]
    zoom = {'medial':1.1, 'lateral':1.2}[aspect]
    p.camera.zoom(zoom)
    if hemi == 'rh':
        p.add_scalar_bar(height=0.5,
                         width=0.1,
                         vertical=True,
                         position_x=0.85,
                         position_y=0.25,
                         color='k',
                         title_font_size=40,
                         label_font_size=30,
                         title='#Electrodes')
        p.update_scalar_bar_range([-100, 100])
    p.show(screenshot=fn+f'_{aspect}.png', auto_close=False)
    axs[i_hemi*2+1].imshow(p.image)
    
    
    fig = ipv.figure(width=500, height=500)
    brain = ipv.plot_trisurf(rr[:, 0], 
                              rr[:, 1], 
                              rr[:, 2],
                              triangles=tris,
                              color=colors)
    
    ipv.squarelim()
    ipv.style.axes_off()
    ipv.style.box_off()
    ipv.pylab.save(fn+'.html')
    fig.close()


print(f'Saved to {fn}')
fn = f'../../../Figures/viz_brain/coverage_{data_type}'
plt.tight_layout()
#plt.subplots_adjust(right=0.85)
plt.savefig(fn+'_all.png')
plt.close(fig_plt)