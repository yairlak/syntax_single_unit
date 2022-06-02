#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:42:06 2022

@author: yl254115
"""

import os
import pickle
import pandas as pd

fn = '/home/yl254115/projects/syntax_single_unit/Output/encoding_models/TRF_patient_479_11_spike_raw_50_50_ridge_None_remove_block in [1,3,5] and word_length>1_position_lexicon_True.pkl'
results_trf, ch_names_trf, args_trf, feature_info_trf = pickle.load(open(fn, 'rb'))


fn = '/home/yl254115/projects/syntax_single_unit/Output/encoding_models/encoding_results_case_study_spike_raw_phonology_decimate_50_smooth_50_patients_479_11_479_25_482_499_502_505_510_513_515_530_538_539_540_541_543_544_549_551.json'
df = pd.read_json(fn)