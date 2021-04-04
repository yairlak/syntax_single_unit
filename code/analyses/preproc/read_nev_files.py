import sys, pickle, argparse, glob
sys.path.append('..')
from utils import load_settings_params
import os.path as op
from neo import io
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--patient', default = '491')
parser.add_argument('--recording-system', choices=['Neuralynx', 'BlackRock'], default='Neuralynx')
args = parser.parse_args()


def check_events(event_nums_zero, time_stamps):
    assert len(event_nums_zero) == len(time_stamps)
    IX = 0
    dict_events = {}
    for block in range(1,7):
        num_non_stimulus_events = 8
        if block in [1, 3, 5]:
            num_events = 508 * 2 + num_non_stimulus_events# 508 word onset + offset
        elif block in [2,4,6]:
            if int(args.patient) > 500:
                num_events = 152 * 2 + num_non_stimulus_events# 152 audio onset + offset of sentence
            else:
                num_events = num_non_stimulus_events
    
        times_events = []
        for i_event in range(num_events):
            next_expected_event = 100 if i_event == 0 else ((i_event-1) % 100) + 1
            curr_event = event_nums_zero[IX]
            while curr_event != next_expected_event:
                IX += 1
                curr_event = event_nums_zero[IX]
            print(block, i_event, IX, curr_event)
            times_events.append(time_stamps[IX])
        dict_events[block] = np.asarray(times_events)
        
    return dict_events


settings = load_settings_params.Settings('patient_' + args.patient)
params = load_settings_params.Params('patient_' + args.patient)
session_folder = op.join(settings.path2patient_folder, 'Raw', 'nev_files')

nev_files = glob.glob(session_folder + '/*.nev')
assert len(nev_files) == 1
nev_file = nev_files[0]

if args.recording_system == 'Neuralynx':
    reader = io.NeuralynxIO(session_folder)
    blks = reader.read(lazy=False)
    #print('Sampling rate of signal:', reader._sigs_sampling_rate)
    time0, timeend = reader.global_t_start, reader.global_t_stop
    internal_event_ids = reader.internal_event_ids
    IX2event_id = {IX:e_id for IX, (x, e_id) in enumerate(internal_event_ids)}
    
    events_times, events_ids = [], []
    for segment in blks[0].segments:
        event_times_mat = segment.events
        for IX, times in enumerate(event_times_mat):
            #if IX2event_id[IX] in event_id.values():
            events_times.extend(times) # in SECONDS
            events_ids.extend([IX2event_id[IX]] * len(times))
    events_ids = np.asarray(events_ids) 
    events_times = np.asarray(events_times)
    IX_chrono = events_times.argsort()
    time_stamps = events_times[IX_chrono]
    event_nums_zero = events_ids[IX_chrono]
    print('time0, timeend = ', time0, timeend)
elif args.recording_system == 'BlackRock':
    reader = io.BlackrockIO(nev_file)
    time0, timeend = reader._seg_t_starts, reader._seg_t_stops
    sfreq = params.sfreq_raw # FROM NOTES
    sfreq = reader.header['unit_channels'][0][-1] # FROM FILE
    events = reader.nev_data['NonNeural'][0]
    time_stamps = [int(e[0]/sfreq) for e in events] 
    event_nums = [e[4] for e in events] 
    event_nums_zero = event_nums - min(event_nums)

plt.plot(time_stamps, event_nums_zero)
plt.show()


dict_events = check_events(event_nums_zero, time_stamps)

print(time0, timeend)

fn = f'events_patient_{args.patient}.pkl'
with open(op.join(session_folder, fn), 'wb') as f:
    pickle.dump([dict_events, time0, timeend], f)
print(f'event times saved to {fn}')         
            