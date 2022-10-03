import os
from nilearn import plotting  
from utils.utils import read_MNI_coord_from_xls

patients = ['553']
data_types = ['micro']
channel_names = ['RAI'] # put None to pick all electrodes
assert len(patients) == len(data_types) == len(channel_names)

dict_colors = {'micro':'r', 'macro':'b'}
path2data = os.path.join('..', '..', 'Data', 'UCLA', 'MNI_coords')

labels, coords, colors = [], [], []
for patient, data_type, channel_name in zip(patients, data_types, channel_names):
    fn_coord = f'sub-{patient}_space-MNI_electrodes.xlsx'
    color = dict_colors[data_type]
    
    df_coords = read_MNI_coord_from_xls(os.path.join(path2data, fn_coord),
                                        data_type, channel_name)
    
    curr_coords = df_coords[['MNI_x', 'MNI_y', 'MNI_z']].values
    curr_labels = df_coords['electrode'].values
    
    labels += list(curr_labels)
    coords += list(curr_coords)
    colors += [color] * len(curr_labels)
    

   

view = plotting.view_markers(coords,
                             colors,
                             marker_labels=labels,
                             marker_size=5) 

view.open_in_browser()
view.save_as_html(f'../../Figures/viz_brain/channel_locations_{"_".join(patients)}_{"_".join(data_types)}_{"_".join(channel_names)}.html')