import os
import pandas as pd
from nilearn import plotting  
from utils.utils import read_MNI_coord_from_xls

subjects_dir = '/volatile/freesurfer/subjects' # your freesurfer directory
patients = '479 482 499 502 504 505 510 513 515 530 538 539 540 541 543 544 545 549 551 552 553 554 556'.split()
path2data = os.path.join('..', '..', 'Data', 'UCLA', 'MNI_coords')
show_labels = False

labels, coords, colors = [], [], []
for patient in patients:
    fn_coord = f'sub-{patient}_space-MNI_electrodes.xlsx'
    for data_type, color in zip(['micro', 'macro'], ['r', 'b']):
        df_coords = read_MNI_coord_from_xls(os.path.join(path2data, fn_coord),
                                            data_type)
        
        curr_coords = df_coords[['MNI_x', 'MNI_y', 'MNI_z']].values
        curr_labels = df_coords['electrode'].values
        
        labels += list(curr_labels)
        coords += list(curr_coords)
        colors += [color] * len(curr_labels)
   

if not show_labels: labels=None
view = plotting.view_markers(coords, colors, marker_labels=labels,
                             marker_size=3) 

view.open_in_browser()
view.save_as_html(f'../../Figures/viz_brain/montage_{"_".join(patients)}.html')



# def get_voxel_roi(atlas_img, coords, labels):
#     """Given a voxel coordinate, return the voxel roi.
#     Arguments:
#         - atlas_img: Niftiimage
#         - coords: tuple (3 values)
#         - labels: list of str
#     Returns:
#         - numpy.array
#     """
#     a, b, c = coords
#     return labels[atlas_img.get_fdata()[a, b, c]]



# # AAL
# aal = datasets.fetch_atlas_aal()
# labels = aal.labels
# atlas = aal.maps
# indices = aal.indices

# atlas_img = load_img(atlas)
# atlas_data = load_img(atlas_img).get_data()
# # labels = np.unique(atlas_data)


# atlas_data = datasets.fetch_atlas_msdl()
# atlas_img = image.concat_imgs((atlas_data['region_coords']))

# get_voxel_roi(atlas_data, [0, 0, 0], atlas_data.labels)



# plotting.pl

# # from nilearn import datasets
# # parcel_dir = '../../resources/rois/'
# # atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011(parcel_dir)
# # atlas_yeo = atlas_yeo_2011['thick_7']
# # from nilearn.regions import connected_label_regions
# # region_labels = connected_label_regions(atlas_yeo)
# # plotting.plot_roi(region_labels,
# # 			cut_coords=(-20,-10,0,10,20,30,40,50,60,70),
# # 			display_mode='z',
# # 			colorbar=True,
# # 			cmap='Paired',
# # 			title='Relabeled Yeo Atlas')
