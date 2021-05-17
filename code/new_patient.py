import os, argparse, glob, shutil

parser = argparse.ArgumentParser()
parser.add_argument('--patient')
parser.add_argument('--session')
parser.add_argument('--copy', action='store_true', default=False)
args = parser.parse_args()

os.makedirs(f'patient_{args.patient}', exist_ok = True)
os.makedirs(f'patient_{args.patient}/Raw/micro/ncs', exist_ok = True)
os.makedirs(f'patient_{args.patient}/Raw/macro/ncs', exist_ok = True)
os.makedirs(f'patient_{args.patient}/Raw/nev_files', exist_ok = True)
os.makedirs(f'patient_{args.patient}/Logs', exist_ok = True)


def find_raw_files(directory, data_type):
    fn_mic, fn_nev = None, None
    if data_type == 'micro':
        fns = glob.glob(directory + '/G??-*.ncs')
        if fns:
            fn_mic = glob.glob(directory + '/MICROPHONE*')
            fn_nev = glob.glob(directory + '/*.nev')
            return fns, fn_mic, fn_nev
        fns = glob.glob(directory + '/*.ns5')
        if fns:
            fn_mic = glob.glob(directory + '/MICROPHONE*')
            fn_nev = glob.glob(directory + '/*.nev')
            return fns, fn_mic, fn_nev
        #print(f"No micro files were found in {directory}")
        return [], fn_mic, fn_nev
    elif data_type == 'macro':
        fns = glob.glob(directory + '/*.ncs')
        fns = [fn for fn in fns if os.path.basename(fn)[3]!='-' and (not os.path.basename(fn).startswith(tuple(['A1', 'A2', 'C3', 'C4', 'EMG', 'EOG', 'Ez', 'Pz', 'MICROPHONE'])))]
        if fns:
            return fns, fn_mic, fn_nev
        fns = glob.glob(directory + '/*.ns3')
        if fns:
            return fns, fn_mic, fn_nev
        #print(f"No micro files were found in {directory}")
        return [], fn_mic, fn_nev
    else:
        raise f"Wrong data type: {data_type}"


for d in os.walk(os.path.join('..', 'Data', 'UCLA', f'{args.patient}_EXP{args.session}')):
    #print(d[0])
    if 'log_patient' in d[0]:
        print('-'*100)
        print('copying logs files:')
        fns_logs = glob.glob(d[0] + '/*.log')
        for fn in fns_logs:
            print(f'copying {fn} to ipatient_{args.patient}/Logs')
            dest = os.path.join('..', 'Data', 'UCLA', f'patient_{args.patient}', 'Logs')
            shutil.copy(fn, dest)
        print('-'*100)
    # MICRO
    fns, fn_mic, fn_nev = find_raw_files(d[0], 'micro')
    if fn_nev:
        for fn in fn_nev:
            print(f'Copying nev file {fn} to patient_{args.patient}/Raw/nev_files')
            shutil.copy(fn, os.path.join('..', 'Data', 'UCLA', f'patient_{args.patient}', 'Raw', 'nev_files'))
        print('-'*100)
    if fn_mic:
        for fn in fn_mic:
            print(f'Copying microphone file {fn} to patient_{args.patient}/Raw')
            shutil.copy(fn, os.path.join('..', 'Data', 'UCLA', f'patient_{args.patient}', 'Raw'))
        print('-'*100)
    if fns:
        print(f'{len(fns)} micro files were found')
        for fn in sorted(fns):
            dest = os.path.join('..', 'Data', 'UCLA', f'patient_{args.patient}', 'Raw', 'micro', 'ncs')
            if args.copy:
                print(f'Copying {fn} to {dest}')
                shutil.copy(fn, dest)
            else:
                print(fn)
        print('-'*100)

# MACRO
    fns, fn_mic, fn_nev = find_raw_files(d[0], 'macro')
    if fns:
        print(f'{len(fns)} macro files were found')
        for fn in sorted(fns):
            dest = os.path.join('..', 'Data', 'UCLA', f'patient_{args.patient}', 'Raw', 'macro', 'ncs')
            if args.copy:
                print(f'Copying {fn} to {dest}')
                shutil.copy(fn, dest)
            else:
                print(fn)
        print('-'*100)
    
