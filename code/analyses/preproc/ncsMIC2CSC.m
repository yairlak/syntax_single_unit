clear; close all; clc;
addpath(genpath('releaseDec2015'))
addpath(genpath('NPMK-5.5.0.0'))

%%
patient = 'patient_499';
recording_system = 'Neuralynx'; % Neuralynx / BlackRock

%% paths
base_folder = ['/neurospin/unicog/protocols/intracranial/syntax_single_unit/Data/UCLA/', patient];
output_path = fullfile(base_folder, 'Raw', 'microphone');

%%
%
switch recording_system
        case 'Neuralynx'
            % Extract raw data and save into MAT files
            ncs_files = dir(fullfile(base_folder, 'Raw', 'microphone', 'MICROPHONE.ncs'));
            
            for ncs_file_name=ncs_files'
                %
                file_name = ncs_file_name.name;
                fprintf('%s\n', ncs_file_name.name)
                ncs_file = fullfile(ncs_file_name.folder, ncs_file_name.name);
                %
                [Timestamps, ChannelNumbers, SampleFrequencies, NumberOfValidSamples, Samples, Header] = Nlx2MatCSC_v3(ncs_file,[1 1 1 1 1],1,1,1);
                data=reshape(Samples,1,size(Samples,1)*size(Samples,2));
                data=int16(data);
                fprintf('%d\n', SampleFrequencies(1))
                samplingInterval = 1000/SampleFrequencies(1);
                save(fullfile(output_path,'MICROPHONE.mat'),'data','samplingInterval', 'file_name');
                fprintf('saved as %s \n', fullfile(output_path,'MICROPHONE.mat'));
            
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
