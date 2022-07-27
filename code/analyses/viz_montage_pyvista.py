#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
import nibabel as nib
import mne
import matplotlib.pyplot as plt
import pandas as pd
import pyvista as pv



def read_MNI_coord_from_xls(fn, data_type):
    ismicro = data_type == 'micro'
    df_coords = pd.read_excel(fn)
    df_coords = df_coords[df_coords['isMicro'] == ismicro]
    coords = [x for x in zip(df_coords['MNI_x'],
                             df_coords['MNI_y'],
                             df_coords['MNI_z'])]
    return dict(zip(df_coords['electrode'], coords))

patient = 'patient_479_11'
data_type = 'micro'
fn_coord = f'sub-{patient[8:11]}_space-MNI_electrodes.xlsx'
path2data = os.path.join('..', '..', 'Data', 'UCLA', patient)
ch_pos = read_MNI_coord_from_xls(os.path.join(path2data, fn_coord),
                                 data_type)

SUBJECTS_DIR = '/volatile/freesurfer/subjects' # your freesurfer directory


labels = mne.read_labels_from_annot(
    'fsaverage', 
    #parc='aparc.a2009s', 
    parc='PALS_B12_Brodmann',
    subjects_dir=SUBJECTS_DIR, 
    verbose=True
)


subjects_dir = SUBJECTS_DIR
hemis=['lh', 'rh']
surface='pial'


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


    colors = curv

    fn = f'../../Figures/viz_brain/montage_{hemi}'
    
    surf = pv.PolyData(vertices, faces)
    surf["colors"] = colors
    p = pv.Plotter(off_screen=True)
    p.set_background('w')
    
    
    p.add_mesh(surf,
               #color='grey',
               show_edges=False,
               scalars="colors",
               cmap='Blues',
               interpolate_before_map=False,
               clim=[0, 1],
               rgb=True)
               #annotations = {chance_level:'chance'})
    
    for ch_name, coords in ch_pos.items():
        p.add_points(np.asarray(coords), render_points_as_spheres=True,
                     point_size=100.0)
     
    #ZY
    p.view_zy()
    p.camera.roll += 90
    aspect = {'lh':'lateral', 'rh':'medial'}[hemi]
    zoom = {'medial':1.65, 'lateral':1.8}[aspect]
    p.camera.zoom(zoom)
    p.show(screenshot = fn + '.png', auto_close=False)
    
    # YZ
#     p.view_yz()
    
    
#     aspect = {'lh':'medial', 'rh':'lateral'}[hemi]
#     #zoom = {'medial':1.65, 'lateral':1.8}[aspect]
#     zoom = {'medial':1.65, 'lateral':1.8}[aspect]
#     p.camera.zoom(zoom)
    
#     p.show(screenshot=fn+f'_{aspect}.png', auto_close=False)
#     x_plt = {'lh':0, 'rh':1}[hemi]
#     y_plt = {'lateral':0, 'medial':1}[aspect]
#     axs[y_plt, x_plt].imshow(p.image)
    
    
#     # ADD COLORBAR
#     if hemi == 'rh':
#         fig_cbar, ax = plt.subplots(1,1,figsize=(2, 2))
#         zoom = 0.1
#         p.camera.zoom(zoom)
#         cbar = p.add_scalar_bar(height=1.4,
#                                 width=1,
#                                 vertical=True,
#                                 position_x=0,
#                                 position_y=0,
#                                 color='k',
#                                 title_font_size=190,
#                                 label_font_size=0,
#                                 fmt='%1.2f',
#                                 title='Brain score')
#         # p.update_scalar_bar_range([0, 2*chance_level])
#         # p.update_scalar_bar_range([0, 1])
#         ax.imshow(p.image)
#         ax.set_axis_off()
#         fn = f'../../../Figures/viz_brain/encoding_{block}_{data_type}_{filt}'
#         plt.subplots_adjust(left=0, right=1)
#         plt.savefig(fn+'_colorbar.png')
#         plt.close(fig_cbar)
#         # p.update_scalar_bar_range([0, 1])
    
#     fig = ipv.figure(width=500, height=500)
#     brain = ipv.plot_trisurf(rr[:, 0], 
#                               rr[:, 1], 
#                               rr[:, 2],
#                               triangles=tris,
#                               color=colors)
    
#     ipv.squarelim()
#     ipv.style.axes_off()
#     ipv.style.box_off()
#     ipv.pylab.save(fn+'.html')
#     fig.close()
    
    
#     print(f'Saved to {fn}')
# fn = f'../../../Figures/viz_brain/encoding_{block}_{data_type}_{filt}'
# plt.figure(fig_plt.number)
# plt.tight_layout()
# #plt.subplots_adjust(right=0.85)
# plt.savefig(fn+'_all.png')
# plt.close(fig_plt)