import mne
from functions import load_settings_params, stats
from functions.utils import probename2picks
import argparse, os
import pandas as pd
import sys

# Set current working directory to that of script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(description='Generate plots')
parser.add_argument('--patient', nargs='*', help='Patient string')
parser.add_argument('--data-type', nargs='*', choices=['micro','macro', 'spike'], default='micro', help='electrode type')
parser.add_argument('--level', nargs='*', choices=['sentence_onset','sentence_offset', 'word', 'phone'], default='sentence_onset', help='')
parser.add_argument('--filter', nargs='*', choices=['raw','gaussian-kernel', 'high-gamma'], default='high-gamma', help='')
parser.add_argument('--thresh', type=float, default=0.05, help='Threshold below which p-value indicates significance')
#
args = parser.parse_args()
args.patient = ['patient_'+p for p in args.patient]
print(args, file=sys.stderr)

for patient in args.patient:
    for data_type in args.data_type:
        for filt in args.filter:
            for level in args.level:
                comparison_name = 'all_end_trials' if level == 'sentence_offset' else 'all_trials'
                settings = load_settings_params.Settings(patient)
                dname_fig = os.path.join(settings.path2figures, 'Comparisons', 'amodal', comparison_name, patient, 'ERPs', data_type)
                if not os.path.exists(dname_fig):
                    os.makedirs(dname_fig)


                fname = '%s_%s_%s_%s-epo' % (patient, data_type, filt, level)
                # READ TEXT FILES (BOTH VISUAL AND AUDITORY) INTO A DATAFRAME
                try:
                    with open(os.path.join(settings.path2epoch_data, fname + '.vis'), 'r') as f:
                        visual_results = f.read().splitlines()
                        visual_results = visual_results[5:] # ignore headers of 4 lines
                    with open(os.path.join(settings.path2epoch_data, fname + '.aud'), 'r') as f:
                        #print(os.path.join(settings.path2epoch_data, fname + '.aud'))
                        auditory_results = f.read().splitlines()
                        auditory_results = auditory_results[5:] # ignore headers of 4 lines
                except:
                    print(f'No responsivness file for {patient} {data_type} {filt} {level}', file=sys.stderr)
                    print(os.path.join(settings.path2epoch_data, fname + '.vis'), file=sys.stderr)
                    continue
                    
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
                    curr_dict = {'patient':patient, 'channel number':channel_number_vis, 'channel name':channel_name_vis, 'p-values_visual':p_vals_vis, 'p-values_auditory':p_vals_aud}
                    df = df.append(curr_dict, ignore_index=True)

                def print_df(df, columns):
                    #df_str = df.to_string(index=False, max_colwidth=40, justify='left', columns=columns)
                    df_str = df.to_string(index=False, justify='left', columns=columns)
                    #df_str = [l.strip().split() for l in df_str.split('\n')]
                    print(df_str)

                df_responsive_both = df[df['p-values_visual'].apply(lambda x: len(x)) > 0] # remove rows with empty lists of p-values
                df_responsive_both = df_responsive_both[df_responsive_both['p-values_auditory'].apply(lambda x: len(x)) > 0] # remove rows with empty lists of p-values
                df_responsive_both = df_responsive_both[(df_responsive_both['p-values_visual'].apply(lambda x: min(x)) < args.thresh) & (df_responsive_both['p-values_auditory'].apply(lambda x: min(x)) < args.thresh)]
                df_responsive_both['p-values_visual'] = df_responsive_both['p-values_visual'].apply(lambda x: [e for e in x if e<args.thresh])
                df_responsive_both['p-values_auditory'] = df_responsive_both['p-values_auditory'].apply(lambda x: [e for e in x if e<args.thresh])
                #print('\nResponsive channels to BOTH presentations (amodal)')
                #print_df(df_responsive_both, None)

                output_log='logs/out_plot_amodal'
                error_log='logs/err_plot_amodal'
                job_name='plot_amodal'
                queue='Nspin_long'
                walltime='02:00:00'
                for index, row in df_responsive_both.iterrows():
                    ch_name = row['channel name']
                    for block_type in ['visual', 'auditory']:
                        fname_fig = 'ERP_trialwise_%s_%s_%s_%s_%s_%s_%s.png' % (patient[8:], data_type, level, filt, ch_name, block_type, comparison_name)
                        fname_fig = os.path.join(dname_fig, fname_fig)
                        #cmd=f'python plot_ERP_trialwise.py --patient {patient[8:]} --data-type {data_type} --level {level} --filter {filt} --block {block_type} --channel-name {ch_name} --comparison-name {comparison_name} --save2 {fname_fig}&'
                        cmd=f'python /neurospin/unicog/protocols/intracranial/Syntax_with_Fried/Code/Main/plot_ERP_trialwise.py --patient {patient[8:]} --data-type {data_type} --level {level} --filter {filt} --block {block_type} --channel-name {ch_name} --comparison-name {comparison_name} --save2 {fname_fig}'
                        print(f'echo {cmd} | qsub -q {queue} -N {job_name} -l walltime={walltime} -o {output_log} -e {error_log}')
