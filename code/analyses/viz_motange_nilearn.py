import os
import mne
import pandas as pd
import numpy as np
from nilearn import plotting  

subjects_dir = '/volatile/freesurfer/subjects' # your freesurfer directory


def read_MNI_coord_from_xls(fn, data_type):
    ismicro = data_type == 'micro'
    df_coords = pd.read_excel(fn)
    df_coords = df_coords[df_coords['isMicro'] == ismicro]
    coords = [x for x in zip(df_coords['MNI_x'],
                             df_coords['MNI_y'],
                             df_coords['MNI_z'])]
    return dict(zip(df_coords['electrode'], coords))


def get_voxel_roi(atlas_img, coords, labels):
    """Given a voxel coordinate, return the voxel roi.
    Arguments:
        - atlas_img: Niftiimage
        - coords: tuple (3 values)
        - labels: list of str
    Returns:
        - numpy.array
    """
    a, b, c = coords
    return labels[atlas_img.get_fdata()[a, b, c]]


from nilearn import datasets
from nilearn.image import load_img

# AAL
aal = datasets.fetch_atlas_aal()
labels = aal.labels
atlas = aal.maps
indices = aal.indices

atlas_img = load_img(atlas)
atlas_data = load_img(atlas_img).get_data()
# labels = np.unique(atlas_data)

from nilearn import datasets

atlas_data = datasets.fetch_atlas_msdl()
from nilearn import image
atlas_img = image.concat_imgs((atlas_data['region_coords']))

get_voxel_roi(atlas_data, [0, 0, 0], atlas_data.labels)

patient = 'patient_479_11'
patient = 'patient_502'
fn_coord = f'sub-{patient[8:11]}_space-MNI_electrodes.xlsx'
path2data = os.path.join('..', '..', 'Data', 'UCLA', patient)

data_type = 'micro'
ch_pos = read_MNI_coord_from_xls(os.path.join(path2data, fn_coord),
                                 data_type)
labels = list(ch_pos.keys())
coords = list(ch_pos.values())
colors = ['r'] * len(coords)
    
data_type = 'macro'
ch_pos = read_MNI_coord_from_xls(os.path.join(path2data, fn_coord),
                                 data_type)
labels += list(ch_pos.keys())
coords += list(ch_pos.values())
colors += ['b'] * len(coords)

view = plotting.view_markers(coords, colors, marker_labels=labels,
                             marker_size=10) 
view.open_in_browser()
view.save_as_html(f'../../Figures/viz_brain/montage_{patient}.html')

plotting.pl

# from nilearn import datasets
# parcel_dir = '../../resources/rois/'
# atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011(parcel_dir)
# atlas_yeo = atlas_yeo_2011['thick_7']
# from nilearn.regions import connected_label_regions
# region_labels = connected_label_regions(atlas_yeo)
# plotting.plot_roi(region_labels,
# 			cut_coords=(-20,-10,0,10,20,30,40,50,60,70),
# 			display_mode='z',
# 			colorbar=True,
# 			cmap='Paired',
# 			title='Relabeled Yeo Atlas')
