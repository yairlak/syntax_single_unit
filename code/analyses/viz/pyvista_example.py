#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:15:22 2021

@author: yl254115
"""

import mne
import numpy as np
import nibabel
import pyvista as pv
import numpy as np

subjects_dir = '/volatile/freesurfer/subjects' # your freesurfer directory

# subjects_dir = '/Users/jeanremi/Desktop/'
mne.datasets.fetch_fsaverage(subjects_dir, verbose=None)
subject = 'fsaverage'
hemi = 'rh'
surface = 'pial'
surface = 'white'
surface = 'inflated_pre'

print(pv.global_theme.camera['viewup'])
# camera = pv.Camera()
fname = subjects_dir + '/%s/surf/%s.%s' % (subject, hemi, surface)
vertices, faces = nibabel.freesurfer.read_geometry(fname)
xmin = vertices[:, 0].min()
xmax = vertices[:, 0].max()
ymin = vertices[:, 1].min()
ymax = vertices[:, 1].max()
zmin = vertices[:, 2].min()
zmax = vertices[:, 2].max()
print(xmin, ymin, zmin)
print(xmax, ymax, zmax)
faces = np.hstack([np.r_[3, i] for i in faces])
surf = pv.PolyData(vertices, faces)
p = pv.Plotter(off_screen=True)
p.add_mesh(surf, color=True, show_edges=False)

# p.view_xy()
# p.view_xy()
# p.set_position([xmin*3, ymin*3, 0])
# p.set_focus([xmax*3, ymax*3, 0])

# print(p.camera.position)
# print(p.camera.focal_point)
# p.camera = camera

# p = pv.Plotter()
p.view_zy()
p.camera.roll += 90
p.camera.zoom = 2
fn = f'test.png'
p.show(screenshot=fn)