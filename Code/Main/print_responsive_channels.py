import mne
from functions import load_settings_params, stats
from functions.utils import probename2picks
import argparse, os
import pandas as pd

# Set current working directory to that of script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(description='Generate plots')
parser.add_argument('--patient', default='479_11', help='Patient string')
parser.add_argument('--data-type', choices=['micro','macro', 'spike'], default='micro', help='electrode type')
parser.add_argument('--level', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='sentence_onset', help='')
parser.add_argument('--filter', choices=['raw','gaussian-kernel', 'high-gamma'], default='high-gamma', help='')
parser.add_argument('--thresh', type=float, default=0.05, help='Threshold below which p-value indicates significance')
#
args = parser.parse_args()
args.patient = 'patient_' + args.patient
print(args)

settings = load_settings_params.Settings(args.patient)
fname = '%s_%s_%s_%s-epo' % (args.patient, args.data_type, args.filter, args.level)

# READ TEXT FILES (BOTH VISUAL AND AUDITORY) INTO A DATAFRAME
with open(os.path.join(settings.path2epoch_data, fname + '.vis'), 'r') as f:
    visual_results = f.read().splitlines()
    visual_results = visual_results[5:] # ignore headers of 4 lines
with open(os.path.join(settings.path2epoch_data, fname + '.aud'), 'r') as f:
    print(os.path.join(settings.path2epoch_data, fname + '.aud'))
    auditory_results = f.read().splitlines()
    auditory_results = auditory_results[5:] # ignore headers of 4 lines
    
df = pd.DataFrame(columns=['channel number', 'channel name', 'p-values_visual', 'p-values_auditory'])
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
    curr_dict = {'channel number':channel_number_vis, 'channel name':channel_name_vis, 'p-values_visual':p_vals_vis, 'p-values_auditory':p_vals_aud}
    df = df.append(curr_dict, ignore_index=True)

def print_df(df, columns):
    df_str = df.to_string(index=False, max_colwidth=40, justify='left', columns=columns)
    #df_str = [l.strip().split() for l in df_str.split('\n')]
    print(df_str)

df_responsive_vis = df[df['p-values_visual'].apply(lambda x: len(x)) > 0] # remove rows with empty lists of p-values
df_responsive_vis = df_responsive_vis[df_responsive_vis['p-values_visual'].apply(lambda x: min(x)) < args.thresh]
df_responsive_vis['p-values_visual'] = df_responsive_vis['p-values_visual'].apply(lambda x: [e for e in x if e<args.thresh])
df_responsive_vis['p-values_auditory'] = df_responsive_vis['p-values_auditory'].apply(lambda x: [e for e in x if e<args.thresh])
print('\nResponsive channels to VISUAL presentation')
print_df(df_responsive_vis, ['channel number', 'channel name', 'p-values_visual'])
df_responsive_aud = df[df['p-values_auditory'].apply(lambda x: len(x)) > 0] # remove rows with empty lists of p-values
df_responsive_aud = df_responsive_aud[df_responsive_aud['p-values_auditory'].apply(lambda x: min(x)) < args.thresh]
df_responsive_aud['p-values_visual'] = df_responsive_aud['p-values_visual'].apply(lambda x: [e for e in x if e<args.thresh])
df_responsive_aud['p-values_auditory'] = df_responsive_aud['p-values_auditory'].apply(lambda x: [e for e in x if e<args.thresh])
print('\nResponsive channels to AUDITORY presentation')
print_df(df_responsive_aud, ['channel number', 'channel name', 'p-values_auditory'])
df_responsive_both = df[df['p-values_visual'].apply(lambda x: len(x)) > 0] # remove rows with empty lists of p-values
df_responsive_both = df_responsive_both[df_responsive_both['p-values_auditory'].apply(lambda x: len(x)) > 0] # remove rows with empty lists of p-values
df_responsive_both = df_responsive_both[(df_responsive_both['p-values_visual'].apply(lambda x: min(x)) < args.thresh) & (df_responsive_both['p-values_auditory'].apply(lambda x: min(x)) < args.thresh)]
df_responsive_both['p-values_visual'] = df_responsive_both['p-values_visual'].apply(lambda x: [e for e in x if e<args.thresh])
df_responsive_both['p-values_auditory'] = df_responsive_both['p-values_auditory'].apply(lambda x: [e for e in x if e<args.thresh])
print('\nResponsive channels to BOTH presentations (amodal)')
print_df(df_responsive_both, None)


