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
parser.add_argument('--block-type', choices=['visual', 'auditory'], default='visual')
parser.add_argument('--thresh', type=float, default=0.05, help='Threshold below which p-value indicates significance')

#
args = parser.parse_args()
args.patients = ['patient_' + p for p in args.patients]
print(args)
# Save2filename
fname2save = f'{args.data_type}_{args.filter}_{args.level}_{args.block_type}'

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
df = pd.DataFrame(columns=['patient', 'channel number', 'channel name', 'p-values'])
for patient in args.patients:
    #print(f'Patient {patient}')
    settings = load_settings_params.Settings(patient)
    fname = '%s_%s_%s_%s-epo' % (patient, args.data_type, args.filter, args.level)
    # READ TEXT FILES (BOTH VISUAL AND AUDITORY) INTO A DATAFRAME
    with open(os.path.join(settings.path2epoch_data, fname + '.' + args.block_type[:3]), 'r') as f:
        results = f.read().splitlines()
        results = results[5:] # ignore headers of 4 lines

    for l in results:
        # Parse text lines
        curr_line = l.strip().split(',')
        channel_number, channel_name = int(curr_line[0]), curr_line[1].strip()
        # parse p-values 
        p_vals = curr_line[2].strip().split(';')
        p_vals = [float(e) for e in p_vals if e]
        # parse cluster times
        cluster_times = curr_line[3].strip().split(';')
        #
        curr_dict = {'patient':patient, 'channel number':channel_number, 'channel name':channel_name, 'p-values':p_vals, 'cluster times':cluster_times}
        df = df.append(curr_dict, ignore_index=True)

print_df(df)

for index, row in df.iterrows():
    probe_name = ''.join([c for c in row['channel name'][4:] if c.isalpha()])
    if not row['p-values']: row['p-values'] = [1]

    first_IX_smaller_than_thresh = [IX for (IX, p) in enumerate(row['p-values']) if p < args.thresh]
    #print(row['p-values'], first_IX_smaller_than_thresh)
    if first_IX_smaller_than_thresh:
        first_IX_smaller_than_thresh = first_IX_smaller_than_thresh[0]
        latency = row['cluster times'][first_IX_smaller_than_thresh].split('-')[0]
    else:
        latency = None
    if latency is None:
        color = 0.5
        marker_size = 1
        IX_no_response.append(cnt)
    else:
        color = float(latency)/1000
        marker_size = 35
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
cax.text(-.01, .5, '0', va='center', ha='right', fontsize=50, transform=cax.transAxes)
cax.text(1.01, .5, '1000 msec', va='center', ha='left', fontsize=50, transform=cax.transAxes)
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))
cax.imshow(gradient, aspect='auto', cmap=cmap)
# Plot brain map
colors = cmap(colors)
for i in IX_no_response:
    colors[i, :] = np.asarray([0., 0., 0., 1])

plotting.plot_connectome(np.identity(len(coords)), np.squeeze(np.asarray(coords)), colors, node_size=marker_sizes, output_file=f"../../Figures/map_latencies_{fname2save}.png", display_mode='lyrz', axes=axes, title=args.level.replace('_', ' '))#, node_kwargs={'edgecolor':colors})  
print(f'Figure saved to: ../../Figures/map_latencies_{fname2save}.png')
matplotlib.pyplot.close(fig)
# GENERATE HTMLs
view = plotting.view_markers(coords, colors, marker_size=np.asarray(marker_sizes)/5) 
view.save_as_html(f"../../Figures/map_latencies_{fname2save}.html")  
print(f'Figure saved to: ../../Figures/map_latencies_{fname2save}.html')
