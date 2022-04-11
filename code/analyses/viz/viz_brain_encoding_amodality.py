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
alpha = 0.05
thresh_r = 0.3 # For visualization only; max value of cbar
data_type = 'spike'
filt = 'raw'
fn_trf_results = f'../../../Output/encoding_models/evoked_encoding_results_decimate_50_smooth_50.json'
df = pd.read_json(fn_trf_results)

df = df.loc[df['data_type'] == data_type]
df = df.loc[df['filter'] == filt]
df = df.loc[df['Feature']=='full']

for block in ['auditory', 'visual']:
    pvals = np.asarray([np.asarray(a) for a in df[f'stats_full_{block}_by_time'].values])
    pvals_cat = np.concatenate(pvals)
    reject_fdr, pvals_fdr = fdr_correction(pvals_cat,
                                           alpha=alpha,
                                           method='indep')
    df[f'pvals_full_{block}_fdr_spatiotemporal'] = pvals_fdr.reshape((pvals.shape[0], -1)).tolist()


# def get_mask_significance(row, block, feature):
#     pvals = row[f'pvals{feature}_{block}_fdr_spatiotemporal']
#     if pvals is None:
#         return None
#     pvals = np.asarray(pvals)
#     reject_fdr, pvals_fdr = fdr_correction(pvals,
#                                            alpha=alpha,
#                                            method='indep')
    
#     return reject_fdr


def get_r_significant(row, block, alpha):
    rs = row[f'r_full_{block}_by_time']
    mask = np.logical_and(np.asarray(row[f'pvals_full_{block}_fdr_spatiotemporal']) <= alpha,
                          np.asarray(row[f'r_full_{block}_by_time']) > 0)
    if mask is None:
        return None
    rs_masked = np.asarray(rs)[mask]
    return rs_masked
    
# for block in ['auditory', 'visual']:
#     for feature in ['', '_full']:
#         df[f'mask_significance_{block}{feature}'] = \
#             df.apply(lambda row: get_mask_significance(row, block, feature),
#                      axis=1)

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


def get_color(x, cmap='seismic'):
    return eval(f'plt.cm.{cmap}(x)')


for block in ['auditory', 'visual']:
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
        
        for label in labels:
            df_roi = df.loc[df['Feature'] == 'full']
            # max_dr = df_roi['dr_auditory_max'].max()
            # print(feature, max_dr)
            df_roi = df_roi.loc[df_roi['ROI_fsaverage'] == label.name]
            #colors[label.vertices, :] = [0,
                                         #df_roi['dr_auditory_max'].mean()/max_dr,
             #                            df_roi['dr_auditory_max'].mean(),
             #                            0]
            # d = df_roi['dr_visual_total'].mean()
            # colors[label.vertices, :] = get_color(df_roi['d_from_diag'].mean())[:3]
            colors[label.vertices, :] = get_color(df_roi[f'r_mean_significant_full_{block}'].mean()/thresh_r)[:3]
            #
        
        # In[14]:
        
        
        fn = f'../../../Figures/viz_brain/encoding_{block}_{hemi}_{data_type}_{filt}'
        
        surf = pv.PolyData(vertices, faces)
        surf["colors"] = colors
        p = pv.Plotter(off_screen=True)
        p.set_background('w')
        
        
        p.add_mesh(surf,
                   #color='grey',
                   show_edges=False,
                   scalars="colors",
                   cmap='seismic',
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
            fig_cbar, ax = plt.subplots(1,1,figsize=(2, 2))
            zoom = 0.1
            p.camera.zoom(zoom)
            cbar = p.add_scalar_bar(height=1.4,
                                    width=1,
                                    vertical=True,
                                    position_x=0,
                                    position_y=0,
                                    color='k',
                                    title_font_size=190,
                                    label_font_size=0,
                                    fmt='%1.2f',
                                    title='Brain score')
            # p.update_scalar_bar_range([0, 2*chance_level])
            # p.update_scalar_bar_range([0, 1])
            ax.imshow(p.image)
            ax.set_axis_off()
            fn = f'../../../Figures/viz_brain/encoding_{block}_{data_type}_{filt}'
            plt.subplots_adjust(left=0, right=1)
            plt.savefig(fn+'_colorbar.png')
            plt.close(fig_cbar)
            # p.update_scalar_bar_range([0, 1])
        
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
    fn = f'../../../Figures/viz_brain/encoding_{block}_{data_type}_{filt}'
    plt.figure(fig_plt.number)
    plt.tight_layout()
    #plt.subplots_adjust(right=0.85)
    plt.savefig(fn+'_all.png')
    plt.close(fig_plt)