#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:46:41 2022

@author: yl254115
"""
import nibabel as nib
import numpy as np
from nilearn import input_data, plotting
from nilearn.image import new_img_like

def voxel_masker(coords_list, img, plot=False, **kwargs_masker): 
    """Returns: - masker for given coordinates - value for input image at this location - voxel coordinates
        coords_list: list
         each element is a tuple for the three coordinates (xyz)
    
    """ 
    if type(img)==str:
        img = nib.load(img)
    
    affine = img.affine[:3, :3] 
    translation = img.affine[:3, 3]
    
    new_data = np.zeros(img.get_fdata().shape)

    for coords in coords_list:
        data_coords = np.matmul(np.linalg.inv(affine), np.array(coords) - translation)
        
        a,b,c = np.apply_along_axis(lambda x: np.round(x, 0), 0, data_coords).astype(int)
        
        value = img.get_fdata()[a, b, c]
        
        new_data[a,b,c] = 1
        
    masker = input_data.NiftiMasker(new_img_like(img, new_data))
    
    masker.set_params(**kwargs_masker)
    
    masker.fit()
    
    if plot: plotting.plot_glass_brain(masker.mask_img, display_mode='lzry')
    
    plotting.show()
    
    return masker, value, [a, b, c]

