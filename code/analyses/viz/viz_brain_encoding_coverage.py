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
data_type = 'spike'
filt = 'raw'
fn_trf_results = f'../../../Output/encoding_models/evoked_encoding_results_decimate_50_smooth_50.json'
df = pd.read_json(fn_trf_results)
df = df.loc[df['data_type'] == data_type]
df = df.loc[df['filter'] == filt]
thresh_num_elec = 30

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
    # print(row['Probe_name'])
    if data_type == 'micro':
        probe_name = row['Probe_name']
    elif data_type=='spike':
        if row['Probe_name'].startswith('G'):
            probe_name = row['Probe_name'].split('-')[1].split('_')[0]
            probe_name = ''.join([c for c in probe_name if not c.isdigit()])
        else:
            probe_name = None
    else:
        probe_name = None
    if probe_name in probename2ROI.keys():
        fsaverage_roi = probename2ROI[probe_name]
    else:
        fsaverage_roi = None
    # print(probe_name)
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
    return eval(f'plt.cm.{cmap}({x})')


fig_plt, axs = plt.subplots(2, 2, figsize=(10, 10))
[ax.set_axis_off() for l_ax in list(axs) for ax in l_ax]

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
    
    
    max_elec = 0
    nums_elec = []
    for label in labels:
        df_roi = df.loc[df['Feature'] == 'full']
        df_roi = df_roi.loc[df_roi['ROI_fsaverage'] == label.name]
        nums_elec.append(len(df_roi))
        if len(df_roi) > max_elec:
            max_elec = len(df_roi)
        colors[label.vertices, :] = get_color(len(df_roi)/thresh_num_elec)[:3]
    
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
    zoom = {'medial':1.65, 'lateral':1.8}[aspect]
    p.camera.zoom(zoom)
    p.show(screenshot=fn+f'_{aspect}.png', auto_close=False)
    x_plt = {'lh':0, 'rh':1}[hemi]
    y_plt = {'lateral':0, 'medial':1}[aspect]
    axs[y_plt, x_plt].imshow(p.image)
    
    
    # YZ
    p.view_yz()
    
    
    aspect = {'lh':'medial', 'rh':'lateral'}[hemi]
    #zoom = {'medial':1.65, 'lateral':1.8}[aspect]
    zoom = {'medial':1.65, 'lateral':1.8}[aspect]
    p.camera.zoom(zoom)
    p.show(screenshot=fn+f'_{aspect}.png', auto_close=False)
    x_plt = {'lh':0, 'rh':1}[hemi]
    y_plt = {'lateral':0, 'medial':1}[aspect]
    axs[y_plt, x_plt].imshow(p.image)
    
    # ADD COLORBAR
    if hemi == 'rh':
        fig_cbar, ax = plt.subplots(1,1,figsize=(5, 2))
        zoom = 0.1
        p.camera.zoom(zoom)
        cbar = p.add_scalar_bar(height=1.4,
                                width=1,
                                vertical=False,
                                position_x=0,
                                position_y=0,
                                color='k',
                                title_font_size=190,
                                label_font_size=0,
                                fmt='%1.2f',
                                title='#elec')
        # p.update_scalar_bar_range([0, 2*chance_level])
        p.update_scalar_bar_range([0, 1])
        ax.imshow(p.image)
        ax.set_axis_off()
        fn = f'../../../Figures/viz_brain/coverage_{data_type}'
        plt.subplots_adjust(left=0, right=1)
        plt.savefig(fn+'_colorbar.png')
        plt.close(fig_cbar)

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
plt.figure(fig_plt.number)
plt.tight_layout()
#plt.subplots_adjust(right=0.85)
plt.savefig(fn+'_all.png')
plt.close(fig_plt)