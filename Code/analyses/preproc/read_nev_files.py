import sys, pickle, argparse, glob
sys.path.append('..')
from utils import load_settings_params
import os.path as op
from neo import io
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--patient', default = '530')
parser.add_argument('--recording-system', choices=['Neuralynx', 'BlackRock'], default='BlackRock')
args = parser.parse_args()


def check_events(event_nums_zero, time_stamps):
    assert len(event_nums_zero) == len(time_stamps)
    IX = 0
    dict_events = {}
    for block in range(1,7):
        num_non_stimulus_events = 7
        if block in [1, 3, 5]:
            num_events = 508 * 2 + num_non_stimulus_events# 508 word onset + offset
        elif block in [2,4,6]:
            num_events = 152 * 2 + num_non_stimulus_events# 152 audio onset + offset of sentence
    
        times_events = []
        next_expected_event = 100
        curr_event = event_nums_zero[IX]
        while curr_event != next_expected_event:
            IX += 1
            curr_event = event_nums_zero[IX]
        times_events.append(time_stamps[IX])
        
        for i_event in range(num_events):
            next_expected_event = (i_event % 100) + 1
            curr_event = event_nums_zero[IX]
            while curr_event != next_expected_event:
                IX += 1
                curr_event = event_nums_zero[IX]
            times_events.append(time_stamps[IX])
        dict_events[block] = np.asarray(times_events[(num_non_stimulus_events+1):])
    return dict_events


settings = load_settings_params.Settings('patient_' + args.patient)
params = load_settings_params.Params('patient_' + args.patient)
sfreq = params.sfreq_raw
session_folder = op.join(settings.path2patient_folder, 'Raw', 'nev_files')

nev_files = glob.glob(session_folder + '/*.nev')
assert len(nev_files) == 1
nev_file = nev_files[0]

if args.recording_system == 'Neuralynx':
    reader = io.NeuralynxIO(session_folder)
    #print('Sampling rate of signal:', reader._sigs_sampling_rate)
    time0, timeend = reader._timestamp_limits[0]
    print('time0, timeend = ', time0, timeend)
elif args.recording_system == 'BlackRock':
    reader = io.BlackrockIO(nev_file)
    events = reader.nev_data['NonNeural'][0]
    time_stamps = [int(1e6*e[0]/sfreq) for e in events] # to microsec
    event_nums = [e[4] for e in events] 
    event_nums_zero = event_nums - min(event_nums)

plt.plot(time_stamps, event_nums_zero)
plt.show()
dict_events = check_events(event_nums_zero, time_stamps)
    
fn = f'events_patient_{args.patient}.pkl'
with open(op.join(session_folder, fn), 'wb') as f:
    pickle.dump(dict_events, f)
print(f'event times saved to {fn}')         
            