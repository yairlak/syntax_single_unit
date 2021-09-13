import mne
import argparse
import os
import matplotlib.pyplot as plt
from utils.data_manip import DataHandler

parser = argparse.ArgumentParser(description='Generate trial-wise plots')
# DATA
parser.add_argument('--patient', default='510', help='Patient string')
parser.add_argument('--data-type', choices=['micro', 'macro', 'spike', 'microphone'],
                    default='micro', help='electrode type')
parser.add_argument('--filter', default='raw', help='')
args = parser.parse_args()

args.patient = 'patient_' + args.patient
print(args)

# LOAD
data = DataHandler(args.patient, args.data_type, args.filter,
                   None, None, None)
# Both neural and feature data into a single raw object
data.load_raw_data(verbose=True)
# GET SENTENCE-LEVEL DATA BEFORE SPLIT
fig = mne.viz.plot_raw_psd(data.raws[0])

