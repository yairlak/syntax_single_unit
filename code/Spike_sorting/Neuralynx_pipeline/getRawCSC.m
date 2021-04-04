clear; close all; clc;

%%
patient = 'patient_493';
% recording_system = 'BlackRock';
recording_system = 'Neuralynx';

%base_folder = ['/home/yl254115/Projects/intracranial/single_unit/Syntax_with_Fried/Data/UCLA/', patient];
base_folder = ['/neurospin/unicog/protocols/intracranial/Syntax_with_Fried/Data/UCLA/', patient];
% base_folder = ['/neurospin/unicog/protocols/intracranial/single_unit_syntax_pipeline/Data/UCLA/', patient];
output_path = fullfile(base_folder, 'Raw', 'macro', 'ChannelsCSC');

mkdir(output_path);
addpath(genpath('releaseDec2015'), genpath('NPMK-4.5.3.0'), genpath('functions'))


%%Get_spikes_CSC_notch2k_ariel_mat
FieldSelection(1) = 1; %     1. Timestamps   
FieldSelection(2) = 1; %     2. Sc Numbers
FieldSelection(3) = 1; %     3. Cell Numbers
FieldSelection(4) = 1; %     4. Params
FieldSelection(5) = 1; %     5. Data Points

ExtractHeader = 0;
ExtractMode = 1;
ModeArray=[]; %all.

%%
%
switch recording_system
        case 'Neuralynx'
            % Extract time0 and timeend from NEV file
            nev_filename = fullfile(base_folder, 'nev_files', 'Events.nev');
%             [TimeStamps, EventIDs, Nttls, Extras, EventStrings] = Nlx2MatEV_v3(nev_filename, FieldSelection, ExtractHeader, ExtractMode, ModeArray);
            
            % Extract raw data and save into MAT files
            ncs_files = dir([base_folder '/Raw/macro/*.ncs']);
            idx=1;
            for ncs_file_name=ncs_files'
                file_name = ncs_file_name.name;
                fprintf('%s\n', file_name)
                ncs_file = fullfile(base_folder,'Raw', 'macro', file_name);
%                 ncs_file = fullfile(base_folder,file_name);
                fprintf('CSC of channnel %d...',idx);
                [Timestamps, ChannelNumbers, SampleFrequencies, NumberOfValidSamples, Samples, Header] = Nlx2MatCSC_v3(ncs_file,[1 1 1 1 1],1,1,1);
                data=reshape(Samples,1,size(Samples,1)*size(Samples,2));
                data=int16(data);
                samplingInterval = 1000/SampleFrequencies(1);
                save(fullfile(output_path,['CSC' num2str(idx) '.mat']),'data','samplingInterval', 'file_name');
                fprintf('saved as %s \n', fullfile(output_path,['CSC' num2str(idx) '.mat']));
                electrodes_info{idx} = ncs_file_name.name;
                idx = idx+1;
            end
        
        case 'BlackRock'
%             nev_file = dir([base_folder '/Raw/*.nev']);
%             nev_file = fullfile(base_folder,'Raw',nev_file(1).name);
%             NEV = openNEV(nev_file, 'read');
%             
            ns5_files = dir([base_folder '/*.ns5']);
            for nc5_file_name=ns5_files'
                file_name = nc5_file_name.name;
                nc5_file = fullfile(base_folder,file_name);
                neuroport2mat_all4(nc5_file, output_path); 
%                 x = openNSx('read',ncs_file);
            end
end
save(fullfile(base_folder, 'electrodes_info_names.mat'), 'electrodes_info')

%% !!! sampling rate !!!! - make sure it's correct
% sr = 40000; 
% channels = 1:(idx-1); %idx=130 for UCLA patient 479
%channels = [13, 47, 48, 49, 55, 57, 59];
%channels = 1:70;
% channels = [62];
not_neuroport = 1;
%% get all csc and produce scs_spikes according to filter and threshold parameters  
%Get_spikes_CSC_notch2k_ariel_mat (channels, fullfile(base_folder, 'ChannelsCSC'), not_neuroport, sr) 

%% only for wave_clus use
%ariel_do_clustering_csc(output_path, channels, sr) 


