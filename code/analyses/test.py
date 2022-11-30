import os
from utils.utils import read_MNI_coord_from_xls

patients = '479 482 499 502 504 505 510 513 515 530 538 539 540 541 543 544 545 549 551 552 553 554 556'.split()
path2data = os.path.join('..', '..', 'Data', 'UCLA', 'MNI_coords')
show_labels = False

labels, coords, colors = [], [], []
for patient in patients:
    fn_coord = f'sub-{patient}_space-MNI_electrodes.xlsx'
    for data_type, color in zip(['micro', 'macro'], ['r', 'b']):
        df_coords = read_MNI_coord_from_xls(os.path.join(path2data, fn_coord),
                                            data_type)

        curr_labels = df_coords['electrode'].values

        print(patient, data_type)
        print(curr_labels)
