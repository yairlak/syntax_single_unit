import os.path as op
from neo import io
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
from utils import load_settings_params
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--patient', default='515')
parser.add_argument('--data-type', default='events', choices=['events', 'micro', 'macro'])
parser.add_argument('--recording-system', default='Neuralynx', choices=['Neuralynx', 'BlackRock'])
args = parser.parse_args()
args.patient = 'patient_' + args.patient

settings = load_settings_params.Settings(args.patient)

if args.data_type == 'events':
    session_folder = op.join(settings.path2patient_folder, 'Raw', 'nev_files')
elif args.data_type == 'micro':
    session_folder = op.join(settings.path2patient_folder, 'Raw', 'micro', 'ncs')
elif args.data_type == 'macro':
    session_folder = op.join(settings.path2patient_folder, 'Raw', 'macro', 'ncs')
print(session_folder)

if args.recording_system == 'Neuralynx':
    NIO = io.NeuralynxIO(session_folder)
    print(NIO)
    if args.data_type != 'events':
        print('Sampling rate of signal:', NIO._sigs_sampling_rate)
    time0, timeend = NIO._timestamp_limits[0]
    print('time0, timeend = ', time0, timeend)
elif args.recording_system == 'BlackRock':
    NIO = io.BlackrockIO(op.join(session_folder, 'Yair_practice_2018Nov09001.nev'))
    events = NIO.nev_data['NonNeural']
    time_stamps = [e[0] for e in events]
    event_num = [e[4] for e in events]
    plt.plot(np.asarray(time_stamps)/40000, event_num, '.')
    plt.show()
