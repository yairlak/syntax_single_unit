import os, sys, argparse
import pandas as pd
import numpy as np
import subprocess
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
from utils.yaml_parser import grid_from_yaml, hierarchize
from utils.data_manip import probe_in_patient

parser = argparse.ArgumentParser()
parser.add_argument('--cluster', action='store_true', default=False)
parser.add_argument('--queue', default='Nspin_long')
parser.add_argument('--walltime', default='72:00:00')
parser.add_argument('--yaml', default=[])
parser.add_argument('--launch', default=False, action='store_true')
parser.add_argument('--all-channels', default=False, action='store_true', help='If True, take all channels and not only responsive ones')
parser.add_argument('--verbose', '-v', action='store_true', default=False)
args = parser.parse_args()

######################
# LOAD DECODING LIST #
######################
# Either from a yaml file or a csv file:
if args.yaml: # generate list from yaml
    decoding_grid = grid_from_yaml(args.yaml)
    decoding_list = hierarchize(decoding_grid) # generates list of dicts with all combinations in yaml
    df_decoding_list = pd.DataFrame(decoding_list)

else: # Load from a csv file (functions/decoding_list.csv)
    df_decoding_list = pd.read_csv('functions/decoding_list.csv', delimiter='\t')
    df_decoding_list = df_decoding_list.where(pd.notnull(df_decoding_list), None) # replace nan with None
if args.verbose:
    print(df_decoding_list.to_string(max_colwidth=30, columns=['comparison-name', 'patient', 'block-type', 'block-type-test', 'filter', 'level', 'probe-name', 'classifier', 'decimate', 'tmin', 'tmax']))


##########################################
# BUILD COMMAND PER ROW IN DECODING LIST #
##########################################
cnt_jobs = 0
for index, row in df_decoding_list.iterrows():
    skip_row = False
    cmd = f"--level {row['level']}" # LEVEL (phone/word/sentence_onset/offset)
    # Loop over PATIENTS/data-types/filters
    patient = row['patient'].split()
    data_type = row['data-type'].split()
    filt = row['filter'].split()
    assert len(patient) == len(data_type) == len(filt)
    for p, d, f in zip(patient, data_type, filt):
        cmd += f" --patient {p} --data-type {d} --filter {f}"
        # PROBE
        if row['probe-name'] is not None and probe_in_patient(row['probe-name'], p): # if None then takes all probes
            probe_name = row['probe-name'].split() # assume syntax from yaml "LSTG-RSTG LMTG" (probes per patient are separated by spaces, and joined with '-')
            if len(probe_name) == 1: # If only one, e.g., "LSTG-RSTG-LMTG-RPMTG" then use this probe for all patients. 
                probes_for_all_patients = probe_name[0].replace('-', ' ')
                #for _ in patient:
                cmd += f" --probe-name {probes_for_all_patients}"
            else: # probes are specified for each patient
                # must have same number of probe groups (e.g., LSTG-RSTG) as the number of patients. 
                assert len(probe_name) == len(patient)
                for p_n in probe_name:
                    cmd += f" --probe-name {p_n.replace('-', ' ')}"
        else:
            cmd += " --probe-name None"
            skip_row = True
    if skip_row:
        continue
    # QUERY
    cmd += f" --comparison-name {row['comparison-name']} --block-type {row['block-type']}"
    if row['fixed-constraint'] is not None:
        cmd += f" --fixed-constraint '{row['fixed-constraint']}'"
    if row['comparison-name-test'] is not None:
        cmd += f" --comparison-name-test {row['comparison-name-test']}"
    if row['block-type-test'] is not None:
        cmd += f" --block-type-test {row['block-type-test']}"
    if row['fixed-constraint-test'] is not None:
        cmd += f" --fixed-constraint-test '{row['fixed-constraint-test']}'"
    # GENERAL
    if row['tmin'] is not None:
        cmd += f" --tmin {row['tmin']}"
    if row['tmax'] is not None:
        cmd += f" --tmax {row['tmax']}"
    if row['vmin'] is not None:
        cmd += f" --vmin {row['vmin']}"
    if row['vmax'] is not None:
        cmd += f" --vmax {row['vmax']}"
    if row['decimate'] is not None:
        cmd += f" --decimate {row['decimate']}"

    if not args.all_channels: #not only responsive ones
        cmd += " --responsive-channels-only"
    # local or cluster
    if not args.cluster: 
        fn_log = f"logs/decoding_{index}.log"
        cmd += f" 2>&1 > {fn_log} &"
        cmd = 'python decode_comparison.py ' + cmd
    else:
        cmd = "python /neurospin/unicog/protocols/intracranial/syntax_single_unit/code/analyses/decoding/decode_comparison.py " + cmd
        output_log=f"logs/decoding_{index}.out"
        error_log=f"logs/decoding_{index}.err"
        job_name=f"decoding_{index}.log"
        
        cmd = f"echo {cmd} | qsub -q {args.queue} -N {job_name} -l walltime={args.walltime} -o {output_log} -e {error_log}"
    # LAUNCH OR PRINT
    if args.launch:
        os.system(cmd) 
    else: # Just ECHO COMMAND
        print(cmd)
    cnt_jobs += 1
print(f"Number of jobs: {cnt_jobs}")
