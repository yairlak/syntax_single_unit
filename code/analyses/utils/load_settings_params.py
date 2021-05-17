import os, glob


class Settings():
    def __init__(self, patient):
        # PATIENT:
        self.hospital = 'UCLA'
        self.patient = patient
        # PATHS
        self.path2patient_folder = os.path.join(
            '..', '..', 'Data', self.hospital, self.patient)
        self.path2log = os.path.join(
            '..', '..', 'Data', self.hospital, self.patient, 'Logs')
        self.path2rawdata = os.path.join(
            '..', '..', 'Data', self.hospital, self.patient, 'Raw')
        self.path2stimuli = os.path.join('..', '..', 'Paradigm')
        self.path2figures = os.path.join('..', '..', 'Figures')
        self.path2output = os.path.join('..', '..', 'Output')

        # Files info
        self.log_name_beginning = \
            'new_with_phones_events_log_in_cheetah_clock_part'
        self.stimuli_file = 'features En_02 sentences.xlsx'
        self.sentences_start_end_filename = 'sentences_start_end_dict.pkl'
        self.stimuli_text_file = 'sentences_Eng_rand_En02.txt'
        self.comparisons_file = 'comparisons.xlsx'
        self.features_file = 'sentence_features.xlsx'
        self.word_features_file = 'word_features.xlsx'
        self.word_features_file2 = 'word_features_new.xlsx'
        self.word2pos_file = 'word2POS.pkl'

        # in MICROSEC
        if self.patient == 'patient_479_11': # Neuralynx
            self.recording_device = 'Neuralynx'
            self.time0 =   1489760586848367 # microsec
            self.timeend = 1489763746079099 # microsec
        if self.patient == 'patient_479_25': # Neuralynx
            self.recording_device = 'Neuralynx'
            self.time0 =   1490191924102607
            self.timeend = 1490194591354836
        if self.patient == 'patient_480': # BlackRock
            self.recording_device = 'BlackRock'
            self.time0 =   0
            self.timeend = 3.313463366666667e+09
        if self.patient == 'patient_482': # Neuralynx
            self.recording_device = 'Neuralynx'
            self.time0 =   1493480044627211
            self.timeend = 1493482901125264
        if self.patient == 'patient_487': # Neuralynx
            self.recording_device = 'Neuralynx'
            self.time0 =   1502557237931999
            self.timeend = 1502560879821412
        if self.patient == 'patient_493': # Neuralynx
            self.recording_device = 'Neuralynx'
            self.time0 =   1520590966262873
            self.timeend = 1520594033849941
        if self.patient == 'patient_502': # Neuralynx
            self.recording_device = 'Neuralynx'
            self.time0 = 1544197879836945
            self.timeend = 1544201196027613
        if self.patient == 'patient_504': # BlackRock
            self.recording_device = 'BlackRock' # MICRO
            self.time0 =   0
            self.timeend = 3.71471593333e+09 # microsec
            self.time0_macro = self.time0 # time zero of MACRO (Neuralynx)
            self.timeend_macro = self.timeend # of MACRO (Neuralynx)
        if self.patient == 'patient_505': # Neuralynx
            self.recording_device = 'Neuralynx'
            self.time0 = 1552403091357879  
            self.timeend = 1552405988561685
        if self.patient == 'patient_510': # BlackRock
            self.recording_device = 'BlackRock'
            self.time0 =   0
            self.timeend = 3.8e+09 # microsec
            self.time0_macro = self.time0 
            self.timeend_macro = self.timeend
        if self.patient == 'patient_513': # Neuralynx
            self.recording_device = 'Neuralynx'
            self.time0 = 1569327112002003  
            self.timeend = 1569329632672960
        if self.patient == 'patient_515': # Neuralynx
            self.recording_device = 'Neuralynx'
            self.time0 = 1572893698748917
            self.timeend = 1572896704058679 
            self.time0_macro = 1572893698749401 # of MACRO (Neuralynx) log files are in neuralynx standard!
            self.timeend_macro = 1572896703932195 # of MACRO (Neuralynx)
        if self.patient == 'patient_530': # BlackRock
            self.recording_device = 'BlackRock' # MICRO
            self.time0 = 0
            self.time0_macro = self.time0 
            self.timeend = 2600*1e6 #

class Params:
    def __init__(self, patient):
        self.patient = patient
        if self.patient == 'patient_479_11': # Neuralynx
            self.sfreq_raw = 40000  # Data sampling frequency [Hz]
            self.sfreq_macro = 40000  # Data sampling frequency [Hz]
        if self.patient == 'patient_479_25': # Neuralynx
            self.sfreq_raw = 40000  # Data sampling frequency [Hz]
            self.sfreq_macro = 40000   # Data sampling frequency [Hz]
        if self.patient == 'patient_480': # BlackRock
            self.sfreq_raw = 0  # Data sampling frequency [Hz]
        if self.patient == 'patient_482': # Neuralynx
            self.sfreq_raw = 40000  # Data sampling frequency [Hz]
            self.sfreq_macro = 40000  # Data sampling frequency [Hz]
        if self.patient == 'patient_487': # Neuralynx
            self.sfreq_raw = 40000  # Data sampling frequency [Hz]
            self.sfreq_macro = 2000  # ??????????? Data sampling frequency [Hz]
        if self.patient == 'patient_489': # Neuralynx
            self.sfreq_raw = 40000  # Data sampling frequency [Hz]
            self.sfreq_macro = -999  # ??????????? Data sampling frequency [Hz]
        if self.patient == 'patient_491': # Neuralynx
            self.sfreq_raw = 40000  # Data sampling frequency [Hz]
            self.sfreq_macro = 2000  # Data sampling frequency [Hz]
        if self.patient == 'patient_493': # Neuralynx
            self.sfreq_raw = 40000  # Data sampling frequency [Hz]
            self.sfreq_macro = 2000  # Data sampling frequency [Hz]
        if self.patient == 'patient_495': # Neuralynx
            self.sfreq_raw = 40000  #???? Data sampling frequency [Hz]
            self.sfreq_macro = 2000  #???? Data sampling frequency [Hz]
        if self.patient == 'patient_496': # Neuralynx
            self.sfreq_raw = 40000  # ?????Data sampling frequency [Hz]
            self.sfreq_macro = 2000  # ??????Data sampling frequency [Hz]
        if self.patient == 'patient_502': # Neuralynx
            self.sfreq_raw = 32000  # Data sampling frequency [Hz]
            self.sfreq_macro = 2000  # Data sampling frequency [Hz]
        if self.patient == 'patient_504': # BlackRock, but Neuralynx for Macro and Microphone
            self.sfreq_raw = 30000  # Data sampling frequency [Hz]
            self.sfreq_macro = 2000  # Data sampling frequency [Hz]
        if self.patient == 'patient_505': # Neuralynx
            self.sfreq_raw = 32000  # Data sampling frequency [Hz]
            self.sfreq_macro = 2000  # Data sampling frequency [Hz]
        if self.patient == 'patient_510': # BlackRock, but Neuralynx for Macro and Microphone
            self.sfreq_raw = 30000  # Data sampling frequency [Hz]
            self.sfreq_macro = 2000  # Data sampling frequency [Hz]
        if self.patient == 'patient_513': # Neuralynx
            self.sfreq_raw = 32000  # ????Data sampling frequency [Hz]
            self.sfreq_macro = 2000  # Data sampling frequency [Hz]
        if self.patient == 'patient_515': # Neuralynx
            self.sfreq_raw = 32000  # ?????Data sampling frequency [Hz]
            self.sfreq_macro = 2000  # Data sampling frequency [Hz]
        if self.patient == 'patient_530': # BlackRock
            self.sfreq_raw = 30000  # Data sampling frequency [Hz]
            self.sfreq_macro = 2000  # Data sampling frequency [Hz]

        self.sfreq_spikes = 1000 # dummy frequency for rasters via MNE [Hz]
        self.line_frequency = 60 # Line frequency [Hz]
        self.tmin = -0.6  # Start time before event [sec], should be negative
        self.tmax = 1.2 # End time after event [sec]
        self.ylim_PSTH = 20 # maximal frequency to present in PSTH [Hz]
        self.downsampling_sfreq = 512

        ###### Frequency bands ##########
        self.iter_freqs = [('High-Gamma', 70, 151, 10)] # (Band-name, min_freq, max_freq, step_freq); the last indicates
        ##################################

        ####### Time-frequency ###########
        self.temporal_resolution = 0.05  # Wavelet's time resolution [sec]
        self.smooth_time_freq = 50 * 1e-3 * self.downsampling_sfreq  # Size of window for Gaussian smoothing the time-freq results. Zero means no smoothing.
        self.smooth_time_freq = 0
        ##################################

        ####### Paradigm  #################
        self.SOA = 500  # [msec]
        self.word_ON_duration = 200 # [msec]
        self.word_OFF_duration = 300  # [msec]
        self.baseline_period = 500 # [msec]
        self.window_st = 50 # [msec] beginning of averaging window used for the vertical plot, relative time 0
        self.window_ed = 250  # [msec] end of averaging window used for the vertical plot, relative to time 0

        if self.baseline_period > abs(self.tmin)*1000:
            import sys
            sys.exit('Basline period must be longer than tmin. Otherwise, baseline cannot be computed.')

        ###################################
