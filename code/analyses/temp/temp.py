import numpy as np
import pandas as pd
from spykes.plot.neurovis import NeuroVis
from spykes.ml.neuropop import NeuroPop
from spykes.io.datasets import load_reaching_data
from spykes.utils import train_test_split
import matplotlib.pyplot as plt

reach_data = load_reaching_data()

print('dataset keys:', reach_data.keys())
print('events:', reach_data['events'].keys())
print('features', reach_data['features'].keys())
print('number of PMd neurons:', len(reach_data['neurons_PMd']))
print('number of M1 neurons:', len(reach_data['neurons_M1']))


neuron_number = 91
spike_times = reach_data['neurons_PMd'][neuron_number - 1]
neuron_PMd = NeuroVis(spike_times, name='PMd %d' % neuron_number)
