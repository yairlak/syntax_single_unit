import os, sys, argparse
import pandas as pd
import numpy as np
import subprocess
from utils.yaml_parser import grid_from_yaml, hierarchize
#from utils.data_manip import probe_in_patient
from utils.utils import get_probe_names

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser()
parser.add_argument('--cluster', action='store_true', default=False)
parser.add_argument('--queue', default='Nspin_long')
parser.add_argument('--walltime', default='72:00:00')
parser.add_argument('--yaml', default=[])
parser.add_argument('--launch', default=False, action='store_true')
parser.add_argument('--all-channels', default=True, action='store_true', help='If True, take all channels and not only responsive ones')
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
    print(df_decoding_list.to_string(max_colwidth=30, columns=['comparison-name', 'patient', 'block-train', 'block-test', 'filter', 'level', 'probe-name', 'classifier', 'decimate', 'tmin', 'tmax']))


def get_probe_list(probe_name, patients, data_types, filts):
    # assume syntax from yaml "LSTG-RSTG LMTG" (probes per patient are separated by spaces, and joined with '-')
    patients_new, data_types_new, filts_new, probe_names = [], [], [], []
    for p, dt, f in zip(patients, data_types, filts):
        if probe_name:
            if probe_name == 'each':
                p_names, _ = get_probe_names(p, dt, '../../Data/UCLA/')
                for p_name in p_names:
                    patients_new.append([p])
                    data_types_new.append([dt])
                    filts_new.append([f])
                    probe_names.append([p_name])
            elif len(probe_name) == 1: # If only one, e.g., "LSTG-RSTG-LMTG-RPMTG" then use this probe for all patients.
                patients_new.append([p])
                data_types_new.append([dt])
                filts_new.append([f])
                probe_names.append([[probe_name[0].replace('-', ' ')]])
            else: # probes are specified for each patient
                patients_new.append([p])
                data_types_new.append([dt])
                filts_new.append([f])
                probe_names.append([probe_name])
        else:
            patients_new.append([p])
            data_types_new.append([dt])
            filts_new.append([f])
            probe_names.append([None])
    
    return patients_new, data_types_new, filts_new, probe_names # lists of lists

##########################################
# BUILD COMMAND PER ROW IN DECODING LIST #
##########################################
cnt_jobs = 0
for index, row in df_decoding_list.iterrows():
    patient = row['patient'].split()
    data_type = row['data-type'].split()
    filt = row['filter'].split()
    assert len(patient) == len(data_type) == len(filt)
    patient_list_of_list, data_type_list_of_list, filt_list_of_list, probe_names_list_of_list = \
            get_probe_list(row['probe-name'], patient, data_type, filt)
    # Each list of list will result in a single command line
    for patient_list, data_type_list, filt_list, probe_names_list in zip(patient_list_of_list,
                                                                         data_type_list_of_list,
                                                                         filt_list_of_list,
                                                                         probe_names_list_of_list):
        #cmd = f"--level {row['level']}" # LEVEL (phone/word/sentence_onset/offset)
        # Loop over PATIENTS/data-types/filters
        
        cmd = f" --comparison-name {row['comparison-name']} --block-train {row['block-train']}"
        if row['fixed-constraint'] is not None:
            cmd += f" --fixed-constraint '{row['fixed-constraint']}'"
        if row['comparison-name-test'] is not None:
            cmd += f" --comparison-name-test {row['comparison-name-test']}"
        if row['block-test'] is not None:
            cmd += f" --block-test {row['block-test']}"
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
        if row['classifier'] is not None:
            cmd += f" --classifier {row['classifier']}"

        if not args.all_channels: #not only responsive ones
            cmd += " --responsive-channels-only"
        for p, d, f, probe_name in zip(patient_list, data_type_list, filt_list, probe_names_list):
            cmd += f" --patient {p} --data-type {d} --filter {f} --probe-name {probe_name}"
            # local or cluster
        if not args.cluster: 
            fn_log = f"logs/decoding_{index}.log"
            cmd += f" 2>&1 > {fn_log} &"
            cmd = 'python decode_comparison.py ' + cmd
        else:
            cmd = "python /neurospin/unicog/protocols/intracranial/syntax_single_unit/code/analyses/decode_comparison.py " + cmd
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
        #raise()
print(f"Number of jobs: {cnt_jobs}")
