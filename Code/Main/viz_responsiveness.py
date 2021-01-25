import os, argparse
import pandas as pd
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from pprint import pprint
from functions import data_manip, load_settings_params
from nilearn import plotting  
import numpy as np
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument('--patients', nargs='*', default=[])
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike'], default='micro', help='electrode type')
parser.add_argument('--level', choices=['sentence_onset', 'sentence_offset', 'word', 'phone'], default='sentence_onset', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'high-gamma'], default='raw', help='')
parser.add_argument('--thresh', type=float, default=0.05, help='Threshold below which p-value indicates significance')

#
args = parser.parse_args()
args.patients = ['patient_' + p for p in args.patients]
print(args)

# Save2filename
fname2save = f'{args.data_type}_{args.filter}_{args.level}'

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

def print_df(df, columns=[]):
    if columns:
        df_str = df.to_string(index=False, max_colwidth=40, justify='left', columns=columns)
    else:
        df_str = df.to_string(index=False, max_colwidth=40, justify='left')
    print(df_str)

# GET AMODALITY PER PATIENT AND PLOT
#dmn_coords_vis, dmn_coords_aud, dmn_coords_both = [], [], []
#colors_vis, colors_aud, colors_both = [], [], []
#marker_sizes_vis, marker_sizes_aud, marker_sizes_both = [], [], []
coords, marker_sizes, colors = [], [], []
cnt = 0
IX_no_response = []
for patient in args.patients:
    #print(f'Patient {patient}')
    settings = load_settings_params.Settings(patient)
    fname = '%s_%s_%s_%s-epo' % (patient, args.data_type, args.filter, args.level)
    # READ TEXT FILES (BOTH VISUAL AND AUDITORY) INTO A DATAFRAME
    with open(os.path.join(settings.path2epoch_data, fname + '.vis'), 'r') as f:
        visual_results = f.read().splitlines()
        visual_results = visual_results[5:] # ignore headers of 4 lines
    with open(os.path.join(settings.path2epoch_data, fname + '.aud'), 'r') as f:
        print(os.path.join(settings.path2epoch_data, fname + '.aud'))
        auditory_results = f.read().splitlines()
        auditory_results = auditory_results[5:] # ignore headers of 4 lines

    df = pd.DataFrame(columns=['patient', 'channel number', 'channel name', 'p-values_visual', 'p-values_auditory'])
    for l_vis, l_aud in zip(visual_results, auditory_results):
        # Parse text lines
        curr_line_vis = l_vis.strip().split(',')
        curr_line_aud = l_aud.strip().split(',')
        channel_number_vis, channel_name_vis = int(curr_line_vis[0]), curr_line_vis[1].strip()
        channel_number_aud, channel_name_aud = int(curr_line_aud[0]), curr_line_aud[1].strip()
        assert channel_number_vis == channel_number_aud and channel_name_vis==channel_name_aud
        # parse p-values from both blocks
        p_vals_vis = curr_line_vis[2].strip().split(';')
        p_vals_vis = [float(e) for e in p_vals_vis if e]
        p_vals_aud = curr_line_aud[2].strip().split(';')
        p_vals_aud = [float(e) for e in p_vals_aud if e]
        #
        curr_dict = {'patient':patient, 'channel number':channel_number_vis, 'channel name':channel_name_vis, 'p-values_visual':p_vals_vis, 'p-values_auditory':p_vals_aud}
        df = df.append(curr_dict, ignore_index=True)

#print_df(df)

for index, row in df.iterrows():
    probe_name = ''.join([c for c in row['channel name'][4:] if not c.isdigit()])
    
    if not row['p-values_visual']: row['p-values_visual'] = [1]
    if not row['p-values_auditory']: row['p-values_auditory'] = [1]

    if min(row['p-values_visual'])<args.thresh:
        p_vis = min(row['p-values_visual'])
        if min(row['p-values_auditory'])<args.thresh:
            p_aud = min(row['p-values_auditory']) 
            color = 0.5 + (p_aud - p_vis)/(args.thresh*4) # color is a value in (0.25, 0.75). Low (high) value means more responsive to auditory (visual) stimuli. 
            marker_size = 35 + 15*(1-min([p_aud, p_vis])/args.thresh)
        else:
            marker_size = 35 + 15*(1- p_vis/args.thresh)
            color = 1 # if significant response only to visual then value is 1. 
    else:
        if min(row['p-values_auditory'])<args.thresh:
            p_aud = min(row['p-values_auditory']) 
            color = 0 # if signifcant response only to audio then value is -1.
            marker_size = 35 + 15*(1- p_aud/args.thresh)
        else:
            color = 0.5 # if no significant response 
            marker_size = 1
            IX_no_response.append(cnt)
    coords.append(elec_locations_dict[probe_name] + 3*np.random.rand(1, 3)) # add some jitter
    colors.append(color)
    marker_sizes.append(marker_size)
    cnt += 1

# Save brain plots
fig , axes = matplotlib.pyplot.subplots(figsize = [30, 10], facecolor='grey')
cmap = matplotlib.cm.get_cmap('viridis') # transform from [0, 1] to the corresponding RGBA in the colormap
# Colorbar
cax = fig.add_axes([0.4, 0.05, 0.2, 0.05])
cax.axis('off')
cax.text(-.01, .5, 'Auditory', va='center', ha='right', fontsize=50, transform=cax.transAxes)
cax.text(1.01, .5, 'Visual', va='center', ha='left', fontsize=50, transform=cax.transAxes)
cax.text(0.5, .5, 'Amodal', va='center', ha='center', fontsize=25, transform=cax.transAxes)
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))
cax.imshow(gradient, aspect='auto', cmap=cmap)
# Plot brain map
colors = cmap(colors)
for i in IX_no_response:
    colors[i, :] = np.asarray([0., 0., 0., 1])

plotting.plot_connectome(np.identity(len(coords)), np.squeeze(np.asarray(coords)), colors, node_size=marker_sizes, output_file=f"../../Figures/map_responsivness_{fname2save}.png", display_mode='lyrz', axes=axes, title=args.level.replace('_', ' '))#, node_kwargs={'edgecolor':colors})  
print(f'Figure saved to: ../../Figures/map_responsivness_{fname2save}.png')
matplotlib.pyplot.close(fig)
# GENERATE HTMLs
view = plotting.view_markers(coords, colors, marker_size=np.asarray(marker_sizes)/5) 
view.save_as_html(f"../../Figures/map_responsivness_{fname2save}.html")  
print(f'Figure saved to: ../../Figures/map_responsivness_{fname2save}.html')
# Electrode locations
plotting.plot_connectome(np.identity(len(coords)), np.squeeze(np.asarray(coords)), np.array([[0,0,0,0.3],]*len(coords)), node_size=3, output_file=f"../../Figures/map_electrode_locations.png", display_mode='lyrz')  
view = plotting.view_markers(coords, 'k', marker_size=2) 
view.save_as_html(f"../../Figures/map_electrode_locations.html")  
print(f'Figure saved to: ../../Figures/map_electrode_locations.png')
print(f'Figure saved to: ../../Figures/map_electrode_locations.html')
