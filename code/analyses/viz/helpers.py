#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 17:24:58 2022

@author: yair
"""
import numpy as np
import nibabel as nib
from nilearn.image import new_img_like

def create_nifti_statmap(coords_list, side_half, values, img, baseline=0): 
    """Returns: - masker for given coordinates - value for input image at this location - voxel coordinates
        coords_list: list
         each element is a tuple for the three coordinates (xyz)
    
    """ 
    if type(img)==str:
        img = nib.load(img)
    
    affine = img.affine[:3, :3] 
    translation = img.affine[:3, 3]
    
    new_data = np.ones(img.get_fdata().shape) * baseline

    for coords, value in zip(coords_list, values):
   
        data_coords_min = np.matmul(np.linalg.inv(affine), np.array(coords-side_half) - translation)
        data_coords_max = np.matmul(np.linalg.inv(affine), np.array(coords+side_half) - translation)
        
        a_min,b_min,c_min = np.apply_along_axis(lambda x: np.round(x, 0), 0, data_coords_min).astype(int)
        a_max,b_max,c_max = np.apply_along_axis(lambda x: np.round(x, 0), 0, data_coords_max).astype(int)
        
        
        for i in range(min(a_min, a_max), max(a_min, a_max)+1):
            for j in range(min(b_min, b_max), max(b_min, b_max)+1):
                for k in range(min(c_min, c_max), max(c_min, c_max)+1):
                   new_data[i,j,k] = value
        
    nimg = new_img_like(img, new_data)
    
   
    
    return nimg
