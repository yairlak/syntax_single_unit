#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 22:30:32 2021

@author: yl254115
"""
import os, glob
import numpy as np
from neo.io import BlackrockIO
import matplotlib.pyplot as plt

path2data = '../../../Data/UCLA/patient_530'


nev_file = glob.glob(os.path.join(path2data, '*.nev'))
assert len(nev_file) == 1
print(nev_file[0])
reader = BlackrockIO(filename=nev_file[0])
blks = reader.read(lazy=False)
TTLs = np.asarray(blks[0].segments[-1].events[0])
diff_TTLs = np.diff(TTLs)
plt.scatter(range(len(TTLs)-1), np.diff(TTLs))

print(TTLs.size)

log_files = glob.glob(os.path.join(path2data, '*.log'))

for fn_log in log_files:
    print(fn_log)
    with open(fn_log, 'r') as f:
        lines = f.readlines()
    times_PTB = [float(l.split()[0]) for l in lines]
    diff_times_PTB = np.diff(times_PTB)
    plt.plot(np.convolve(diff_TTLs, diff_times_PTB, 'same'))
    print(len(times_PTB))
    #plt.scatter(range(len(times_PTB)-1), np.diff(times_PTB))