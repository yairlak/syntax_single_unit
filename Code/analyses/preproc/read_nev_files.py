from functions import load_settings_params
import os.path as op
from neo import io
import matplotlib.pyplot as plt
import numpy as np
import sys

recording_system = 'Neuralynx'
settings = load_settings_params.Settings(sys.argv[1])
session_folder = op.join(settings.path2patient_folder, 'Raw', 'nev_files')

#session_folder = op.join('../../Data/UCLA/patient_479_25', 'Raw', 'micro', 'ncs')
# settings = load_settings_params.Settings('patient_504')
# session_folder = '/nfs/neurospin/unicog/protocols/intracranial/Syntax_with_Fried/Data/UCLA/patient_504/Raw/macro/ncs/'
# settings = load_settings_params.Settings(sys.argv[1])
# session_folder = op.join(settings.path2patient_folder, 'Raw', 'nev_files')
#session_folder = '/neurospin/unicog/protocols/intracranial/Syntax_with_Fried/Data/UCLA/patient_504/Raw/macro/ncs'
print(session_folder)

if recording_system == 'Neuralynx':
    NIO = io.NeuralynxIO(session_folder)
    print(NIO)
    #print('Sampling rate of signal:', NIO._sigs_sampling_rate)
    time0, timeend = NIO._timestamp_limits[0]
    print('time0, timeend = ', time0, timeend)
elif recording_system == 'BlackRock':
    NIO = io.BlackrockIO(op.join(session_folder, 'Yair_practice_2018Nov09001.nev'))
    events = NIO.nev_data['NonNeural']
    time_stamps = [e[0] for e in events]
    event_num = [e[4] for e in events]
    plt.plot(np.asarray(time_stamps)/40000, event_num, '.')
    plt.show()
