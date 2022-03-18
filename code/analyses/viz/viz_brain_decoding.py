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


# In[2]:
data_type_filters = ['micro_raw', 'micro_high-gamma', 'spike_raw']
# comparison_names = ['number', 'dec_quest_len2', 'embedding_vs_long',
#                     'pos_simple', 'word_string_first']
comparison_names = ['pos_simple']
hemis = ['lh', 'rh']
block_types = ['visual', 'auditory']
block_types = ['visual', 'auditory', 'amodal_intersection', 'amodal_union']

# In[3]:
SUBJECTS_DIR = '/volatile/freesurfer/subjects' # your freesurfer directory
#SUBJECTS_DIR = '../../../../freesurfer/subjects' # your freesurfer directory


fn_results = f'../../../Output/decoding/decoding_results_decimate_50_smooth_50.json'
df = pd.read_json(fn_results)

# In[4]:
# FDR correction
alpha = 0.05


# In[5]:
for data_type_filter in data_type_filters:
    for comparison_name in comparison_names:
        print(data_type_filter.upper(), comparison_name.upper())
        for block_type in block_types:
            # for block_test in block_types:
            fig_plt, axs = plt.subplots(2, 2, figsize=(10, 10)) #gridspec_kw={'height_ratios': [3, 3, 1]})
            [ax.set_axis_off() for l_ax in list(axs) for ax in l_ax]
            
            # RETRIEVE RELEVANT DATA
            df_whole_brain = df.loc[df['data-type_filters'] == data_type_filter]
            df_whole_brain = df_whole_brain.loc[\
                df_whole_brain['comparison_name'] == comparison_name]
            if block_type.startswith('amodal'):
                df_whole_brain = df_whole_brain.loc[\
                    df_whole_brain['block_train'] != df_whole_brain['block_test']]
            else:
                df_whole_brain = df_whole_brain.loc[\
                    df_whole_brain['block_train'] == block_type]
                df_whole_brain = df_whole_brain.loc[\
                        df_whole_brain['block_test'] == block_type]                
            if df_whole_brain.empty:
                continue
            
            # FDR CORRECTION AT THE BRAIN LEVEL
            pvals = df_whole_brain['pvals'].values # n_ROIs X n_times
            pvals_cat = np.concatenate(pvals)
            reject_fdr, pvals_fdr = fdr_correction(pvals_cat,
                                                   alpha=alpha,
                                                   method='indep')
            df_whole_brain['pvals_fdr_whole_brain'] = pvals_fdr.reshape((pvals.shape[0], -1)).tolist()
            df_whole_brain['reject_fdr_whole_brain'] = reject_fdr.reshape((pvals.shape[0], -1)).tolist()

            for i_hemi, hemi in enumerate(hemis):
                
                                    
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
                        scores = df_roi['scores'].values
                        scores = np.asarray([np.asarray(l) for l in scores])
                        reject_fdr = df_roi['reject_fdr_whole_brain'].values
                        reject_fdr = np.asarray([np.asarray(l) for l in reject_fdr]).squeeze()
                        
                        # MASK by FDR and CHANCE
                        if block_type == 'amodal_intersection':
                            reject_fdr = reject_fdr.all(axis=0)
                        elif block_type == 'amodal_union':
                            reject_fdr = reject_fdr.any(axis=0)
                        
                        # chance_level = df_roi['chance_level'].values[0]
                        chance_level = 0.5
                        scores_masked = scores.copy()
                        scores_masked[scores_masked<chance_level] = np.nan
                        scores_sig = scores_masked[:, reject_fdr]
                        
                        # average across time (and if amodal, also across V2A and A2V)
                        if scores_sig.size>0:
                            scores_sig_mean = np.nanmean(scores_sig)
                        
                        
                        # if (not np.isnan(scores_sig_mean)):
                            # if scores_sig.mean(axis=0) > chance_level:
                            colors[label.vertices, :] = plt.cm.RdBu_r(scores_sig_mean/chance_level/2)[:3]
                                
                        
                    
                    # In[14]:
                    
                fn = f'../../../Figures/viz_brain/decoding_{comparison_name}_{hemi}_{block_type}_{data_type_filter}'
                
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
                           # clim=[0, 2*chance_level],
                           clim=[0.2, 0.8],
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
                x_plt = {'lh':0, 'rh':1}[hemi]
                y_plt = {'lateral':0, 'medial':1}[aspect]
                axs[y_plt, x_plt].imshow(p.image)
                
                # YZ
                p.view_yz()
                
                
                aspect = {'lh':'medial', 'rh':'lateral'}[hemi]
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
                                            title='AUC')
                    # p.update_scalar_bar_range([0, 2*chance_level])
                    p.update_scalar_bar_range([0.2, 0.8])
                    ax.imshow(p.image)
                    ax.set_axis_off()
                    fn = f'../../../Figures/viz_brain/decoding_{comparison_name}_{block_type}_{data_type_filter}'
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
            fn = f'../../../Figures/viz_brain/decoding_{comparison_name}_{block_type}_{data_type_filter}'
            plt.figure(fig_plt.number)
            plt.tight_layout()
            # plt.subplots_adjust(hspace=10)
            plt.savefig(fn+'_all.png')
            plt.close(fig_plt)
            
                # In[ ]:
