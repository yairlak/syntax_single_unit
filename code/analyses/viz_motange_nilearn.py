import os
import pandas as pd
from nilearn import plotting  
from utils.utils import read_MNI_coord_from_xls
import nibabel as nib
from utils.viz import voxel_masker
from nilearn import surface
from nilearn import datasets
from nilearn import plotting
from nilearn import image

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
   

fn = f'../../Figures/viz_brain/montage_{"_".join(patients)}'

if not show_labels: labels=None
view = plotting.view_markers(coords, colors, marker_labels=labels,
                             marker_size=3) 

# view.open_in_browser()
view.save_as_html(fn + '.html')

# MASKER

mask_ICV = '../../../templates/mask_ICV.nii'
fsaverage = datasets.fetch_surf_fsaverage()
img_ICV = nib.load(mask_ICV)

masker, value, [a, b, c] = voxel_masker(coords, img_ICV, plot=False)

# SMOOTH
mask_img = masker.mask_img
mask_img = image.smooth_img(mask_img, 8.)

# GLASS BRAIN
fig_glass = plotting.plot_glass_brain(mask_img, display_mode='lzr')
fig_glass.savefig(fn + '_glass.png')

# PROJECT ONTO THE SURFACE
texture = surface.vol_to_surf(mask_img, fsaverage.pial_right)
fig = plotting.plot_surf_stat_map(
    fsaverage.infl_right, texture, hemi='right',
    title='Surface right hemisphere', colorbar=True,
    threshold=0, bg_map=fsaverage.sulc_right
)
texture = surface.vol_to_surf(mask_img, fsaverage.pial_left)
fig = plotting.plot_surf_stat_map(
    fsaverage.infl_left, texture, hemi='left',
    title='Surface right hemisphere', colorbar=True,
    threshold=0, bg_map=fsaverage.sulc_right
)

fig_surf, axs = plotting.plot_img_on_surf(mask_img, 
                          views=['lateral', 'medial'],
                          hemispheres=['left', 'right'],
                          colorbar=True,
                          threshold=1e-5)

# PROJECT ONTO THE 
for inflate in [False, True]:
    fig_surf, axs = plotting.plot_img_on_surf(mask_img,
                                              surf_mesh = fsaverage,
                                              views=['lateral', 'medial'],
                                              hemispheres=['left', 'right'],
                                              inflate=inflate,
                                              colorbar=True,
                                              threshold=1e-5)
    fig_surf.savefig(fn + f'_surf_inflate_{inflate}.png')

# fig.show()
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
