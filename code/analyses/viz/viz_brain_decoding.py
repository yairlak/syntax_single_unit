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

print(pv.__version__)


# In[2]:
data_type_filters = ['spike_raw']
comparison_names = ['number', 'dec_quest_len2', 'embedding_vs_long',
                    'pos_simple', 'word_string_first']
# comparison_names = ['word_string_first']
hemis = ['lh', 'rh']
block_types = ['visual', 'auditory']

# In[3]:
SUBJECTS_DIR = '/volatile/freesurfer/subjects' # your freesurfer directory


fn_results = f'../../../Output/decoding/decoding_results.json'
df = pd.read_json(fn_results)

# In[4]:
# FDR correction
alpha = 0.05
pvals = df['pvals'].values # n_ROIs X n_times
pvals_cat = np.concatenate(pvals)
reject_fdr, pvals_fdr = fdr_correction(pvals_cat,
                                       alpha=alpha,
                                       method='indep')
df['pvals_fdr_whole_brain'] = pvals_fdr.reshape((pvals.shape[0], -1)).tolist()
df['reject_fdr_whole_brain'] = reject_fdr.reshape((pvals.shape[0], -1)).tolist()


# In[5]:
for data_type_filter in data_type_filters:
    for comparison_name in comparison_names:
        print(data_type_filter.upper(), comparison_name.upper())
        for block_train in block_types:
            for block_test in block_types:
                fig_plt, axs = plt.subplots(1, 4, figsize=(20, 5))
                [ax.set_axis_off() for ax in axs]
                for i_hemi, hemi in enumerate(hemis):
                    # RETRIEVE RELEVANT DATA
                    df_whole_brain = df.loc[df['data-type_filters'] == data_type_filter]
                    df_whole_brain = df_whole_brain.loc[\
                        df_whole_brain['comparison_name'] == comparison_name]
                    df_whole_brain = df_whole_brain.loc[\
                        df_whole_brain['block_train'] == block_train]
                    df_whole_brain = df_whole_brain.loc[\
                            df_whole_brain['block_test'] == block_test]                
                    if df_whole_brain.empty:
                        continue

                                        
                    # In[6]:
                    ROI_fsaverages = list(df['ROI'])
                    ROI_fsaverages = list(set([roi for roi in ROI_fsaverages \
                                               if roi is not None]))
                    
                    # In[7]:
                    labels = mne.read_labels_from_annot(
                        'fsaverage', 
                        #parc='aparc.a2009s', 
                        parc='PALS_B12_Brodmann',
                        subjects_dir=SUBJECTS_DIR, 
                        verbose=True
                    )
                    
                    
                    labels = [label for label in labels \
                              if label.name in ROI_fsaverages]
                    
                    # In[8]:
                    
                    
                    # def read_surface(subjects_dir, hemi='lh', surface='pial'):
                    subjects_dir = SUBJECTS_DIR
                    surface='pial'
                    # Read Anatomy
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
                    
                    
                    # In[9]:
                    n_areas = len(ROI_fsaverages)
                    X, Y = np.meshgrid(np.linspace(0, 1, int(n_areas**.5)),
                                       np.linspace(0, 1, int(n_areas**.5)),
                                       sparse=False, indexing='xy')
                    
                    
                    # In[10]:
                    colors = curv
                    
                    def get_color(x, cmap='RdBu_r'):
                        return eval(f'plt.cm.{cmap}((np.clip(x,-1,1)+1)/2.)')
                    
                    for label in labels:
                        
                        df_roi = df_whole_brain.loc[df_whole_brain['ROI'] == label.name]
                        if not df_roi.empty:
                            scores = np.asarray(df_roi['scores'].values[0])
                            scores_mean = scores.mean(axis=0)
                            reject_fdr = df_roi['reject_fdr_whole_brain'].values[0]
                            
                            scores_mean_sig = scores_mean[reject_fdr]
                            chance_level = df_roi['chance_level'].values[0]
                            if scores_mean_sig.shape[0]>0:
                                colors[label.vertices, :] = plt.cm.RdBu_r(scores_mean_sig.max(axis=0)/chance_level/2)[:3]
                            
                        
                        # In[14]:
                        
                    fn = f'../../../Figures/viz_brain/decoding_{comparison_name}_{hemi}_{block_train}_{block_test}_{data_type_filter}'
                    
                    surf = pv.PolyData(vertices, faces)
                    surf["colors"] = colors
                    p = pv.Plotter(off_screen=True)
                    p.set_background('w')
                    
                    
                    p.add_mesh(surf,
                               color='grey',
                               show_edges=False,
                               scalars="colors",
                               cmap='RdBu_r',
                               interpolate_before_map=False,
                               clim=[0, 2*chance_level],
                               rgb=True,
                               annotations = {chance_level:'chance'})
                    
                    # https://docs.pyvista.org/examples/02-plot/scalar-bars.html
                        # p.export_html('test.html')
                    
                    #ZY
                    p.view_zy()
                    p.camera.roll += 90
                    aspect = {'lh':'lateral', 'rh':'medial'}[hemi]
                    zoom = {'medial':1.65, 'lateral':1.8}[aspect]
                    p.camera.zoom(zoom)
                    p.show(screenshot=fn+f'_{aspect}.png', auto_close=False)
                    axs[i_hemi*2].imshow(p.image)
                    
                    # YZ
                    p.view_yz()
                    
                    
                    aspect = {'lh':'medial', 'rh':'lateral'}[hemi]
                    zoom = {'medial':1.65, 'lateral':1.8}[aspect]
                    p.camera.zoom(zoom)
                    if hemi == 'rh':
                        p.add_scalar_bar(height=0.25,
                                         vertical=True,
                                         position_x=0.9,
                                         position_y=0.74,
                                         color='k',
                                         title_font_size=20,
                                         label_font_size=16,
                                         title='Accuracy')
                        p.update_scalar_bar_range([0, 2*chance_level])
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
                fn = f'../../../Figures/viz_brain/decoding_{comparison_name}_{block_train}_{block_test}_{data_type_filter}'
                plt.tight_layout()
                plt.savefig(fn+'_all.png')
                plt.close(fig_plt)
                    # In[ ]:
                    
                    
                    
                    
