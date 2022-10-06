import os
import mne
import pandas as pd
import numpy as np

subjects_dir = '/volatile/freesurfer/subjects' # your freesurfer directory


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
fn_coord = 'sub-patient_479_space-MNI_electrodes.xlsx'
path2data = os.path.join('..', '..', 'Data', 'UCLA', patient)
ch_pos = read_MNI_coord_from_xls(os.path.join(path2data, fn_coord),
                                 data_type)

montage = mne.channels.make_dig_montage(ch_pos, coord_frame='head')

trans = mne.transforms.Transform(fro='head', to='mri',
                                             trans=np.eye(4))

montage.apply_trans(trans)
#trans = mne.channels.compute_native_head_t(montage)

info = mne.create_info(list(ch_pos.keys()), sfreq=1000)
info.set_montage(montage)

#fig = mne.viz.plot_alignment(info, trans,
#                             'fsaverage', subjects_dir=subjects_dir,
#                             surfaces=['pial'], coord_frame='mri')

#fig.plotter.show()
#fig._plotter.show(screenshot = 'test.png', auto_close=False)
#brain = mne.viz.Brain('fsaverage', alpha=0.1, cortex='low_contrast',
#                      subjects_dir=subjects_dir, units='m', figure=fig)

#mne.viz.plot_montage(montage, show_names=False, sphere=1)

fig = mne.viz.plot_alignment(info, trans, 'fsaverage',
                             subjects_dir=subjects_dir, show_axes=True,
                             surfaces=['pial', 'head'], coord_frame='mri')


print(dir(fig))

#p.show(screenshot = 'test.png', auto_close=False)
