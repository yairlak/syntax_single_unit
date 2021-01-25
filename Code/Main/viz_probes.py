import os, argparse
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from pprint import pprint
from functions import data_manip
from nilearn import plotting  
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--patients', nargs='*', default=[])
args=parser.parse_args()

patients = ['patient_' + p for p in args.patients]


patient_colors = {}
patient_colors['patient_479_11'] = 'b'
patient_colors['patient_479_25'] = 'b'
patient_colors['patient_482'] = 'r'
patient_colors['patient_487'] = 'g'
patient_colors['patient_493'] = 'c'
patient_colors['patient_502'] = 'm'
patient_colors['patient_504'] = 'aqua'
patient_colors['patient_505'] = 'y'
patient_colors['patient_510'] = 'k'
patient_colors['patient_513'] = 'g'
patient_colors['patient_515'] = 'orange'

#print(patient_colors)

def get_names(patient, micro_macro):
    print(patient, micro_macro)
    path2channel_names = os.path.join('..', '..', 'Data', 'UCLA', patient, 'Raw', micro_macro, 'CSC_mat', 'channel_numbers_to_names.txt')
    try:
        with open(path2channel_names, 'r') as f:
            channel_names = f.readlines()
        channel_names = [l.strip().split('\t')[1] for l in channel_names]
        if micro_macro == 'micro':
            channel_names.pop(0) # remove MICROPHONE line
            channel_names = [s[4::] for s in channel_names] # remove prefix if exists (in micro: GA1-, GA2-, etc)
        channel_names = [s[:-5] for s in channel_names] # remove file extension and electrode numbering (e.g., LSTG1, LSTG2, LSTG3) 
        if (micro_macro == 'macro') & (patient == 'patient_502'):
            channel_names = [name for name in channel_names if name not in ['ROF', 'RAF']] # 502 has more macro than micro see Notes/log_summary.txt (March 2020)
            print('Macros also include ROF and RAF - see Notes/log_summary.txt (2020Mar02)')
    except:
        print('!!! - Missing %s channel-name files for %s' % (micro_macro, patient))
        return
    return sorted(list(set(channel_names)))


# MAIN

# Get elec locations in MNI
with open('../../Data/UCLA/MNI_coordinates.txt') as f:
    elec_locations = f.readlines()

elec_locations_dict = {}
for l in elec_locations:
    loc = l.split(',')[0]
    x = float(l.split(',')[1])
    y = float(l.split(',')[2])
    z = float(l.split(',')[3])
    elec_locations_dict[loc] = (x, y, z)
#print(elec_locations_dict)

# Get probe names per patient
names_from_all_patients = []
dmn_coords = []
colors = []
for patient in patients:
    names_micro = get_names(patient, 'micro')
    for _ in range(8):
        for probe_name in names_micro:
            #print(probe_name)
            if probe_name in elec_locations_dict.keys():
                coords = elec_locations_dict[probe_name] + 3*np.random.rand(1, 3) # add some jitter
                dmn_coords.append(coords)
                colors.append(patient_colors[patient])
            else:
                print('probe name not in elec location list: %s %s:' % (probe_name, patient))

#print(dmn_coords, colors)
view = plotting.view_markers(dmn_coords, colors, marker_size=3) 
view.save_as_html("../../Output/HTMLs/elec_locations_plot.html")  
print("HTML saved to: ../../Output/HTMLs/elec_locations_plot.html")
