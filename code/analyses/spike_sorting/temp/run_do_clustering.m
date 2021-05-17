clear; close all; clc;

%%
patient = 'patient_479';

%base_folder = ['/home/yl254115/Projects/single_unit_syntax/Data/UCLA/', patient];
% base_folder = ['/neurospin/unicog/protocols/intracranial/single_unit_syntax_pipeline/Data/UCLA/', patient, '/Macro'];
base_folder = ['/neurospin/unicog/protocols/intracranial/single_unit_syntax_pipeline/Data/UCLA/', patient];
% base_folder = ['A:/protocols/intracranial/single_unit_syntax_pipeline/Data/UCLA/', patient];
output_path = fullfile(base_folder,'ChannelsCSC');


% mkdir(output_path);
addpath(genpath('releaseDec2015'), genpath('NPMK-4.5.3.0'), genpath('functions'))

%% !!! sampling rate !!!! - make sure it's correct
sr = 40000; 
% channels = 1:(idx-1); %idx=130 for UCLA patient 479
channels = 6:70;
% channels = [17:18];
% channels = [62];
not_neuroport = 1;

%% only for wave_clus use
ariel_do_clustering_csc(output_path, channels, sr) 


