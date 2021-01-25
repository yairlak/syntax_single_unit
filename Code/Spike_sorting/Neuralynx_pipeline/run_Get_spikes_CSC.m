clear; close all; clc;

%%
<<<<<<< HEAD
<<<<<<< HEAD
patient = 'patient_479';

%base_folder = ['/home/yl254115/Projects/single_unit_syntax/Data/UCLA/', patient];
base_folder = ['/neurospin/unicog/protocols/intracranial/single_unit_syntax_pipeline/Data/UCLA/', patient];
=======
patient = 'patient_493';

base_folder = ['/home/yl254115/Projects/single_unit_syntax/Data/UCLA/', patient];
% base_folder = ['/neurospin/unicog/protocols/intracranial/single_unit_syntax/Data/UCLA/', patient];
>>>>>>> 53fa704ad0f6f69e3e0afe860395a33a4a700590
=======
patient = 'patient_487';

%base_folder = ['/home/yl254115/Projects/single_unit_syntax/Data/UCLA/', patient];
base_folder = ['/neurospin/unicog/protocols/intracranial/single_unit_syntax_pipeline/Data/UCLA/', patient];
>>>>>>> 52d67cfe233746963a7f9004577b5f7b98ab4e7e
output_path = fullfile(base_folder,'ChannelsCSC');

% mkdir(output_path);
addpath(genpath('releaseDec2015'), genpath('NPMK-4.5.3.0'), genpath('functions'))
rmpath(genpath(fullfile('..', 'wave_clus-testing')))

%% !!! sampling rate !!!! - make sure it's correct
sr = 40000; 
<<<<<<< HEAD
<<<<<<< HEAD
channels = 67:70;
=======
channels = 1:10;
>>>>>>> 53fa704ad0f6f69e3e0afe860395a33a4a700590
=======
channels = 1:80;
>>>>>>> 52d67cfe233746963a7f9004577b5f7b98ab4e7e
%channels = [26, 28, 51];
% channels = [17:18];
not_neuroport = 1;

%% get all csc and produce scs_spikes according to filter and threshold parameters  
Get_spikes_CSC_notch2k_ariel_mat(channels, fullfile(base_folder, 'ChannelsCSC'), not_neuroport, sr) 
