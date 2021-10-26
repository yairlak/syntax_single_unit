clear; close all; clc;
addpath(genpath('releaseDec2015'), genpath('NPMK-5.5.0.0'), genpath('functions'))

%%
patient = 'patient_544';
elec_type = 'micro'; % micro / macro
recording_system = 'BlackRock'; % Neuralynx / BlackRock

%% pathsls 
base_folder = ['/neurospin/unicog/protocols/intracranial/syntax_single_unit/Data/UCLA/', patient];
output_path = fullfile(base_folder, 'Raw', elec_type, 'mat');
%mkdir(output_path);

%%
%
fid = fopen(fullfile(output_path, 'sfreq.txt'),'wt');
switch recording_system
        case 'Neuralynx'
            % Extract time0 and timeend from NEV file
            nev_filename = fullfile(base_folder, 'nev_files', 'Events.nev');
            
            % Extract raw data and save into MAT files
            ncs_files = dir(fullfile(base_folder, 'Raw', elec_type, 'ncs', '*.ncs'));
            idx=1;
            for ncs_file_name=ncs_files'
                fprintf('CSC of channnel %d...',idx);
                %
                file_name = ncs_file_name.name;
                fprintf('%s\n', ncs_file_name.name)
                ncs_file = fullfile(fullfile(base_folder, 'Raw', elec_type, 'ncs'), ncs_file_name.name);
                %
                [Timestamps, ChannelNumbers, SampleFrequencies, NumberOfValidSamples, Samples, Header] = Nlx2MatCSC_v3(ncs_file,[1 1 1 1 1],1,1,1);
                data=reshape(Samples,1,size(Samples,1)*size(Samples,2));
                data=int16(data);
                samplingInterval = 1000/SampleFrequencies(1);
                fprintf('Sampling freq:')
                fprintf('%i\n', SampleFrequencies(1))
                fprintf(fid, num2str(SampleFrequencies(1)))
                save(fullfile(output_path,['CSC' num2str(idx) '.mat']),'data','samplingInterval', 'file_name');
                fprintf('saved as %s \n', fullfile(output_path,['CSC' num2str(idx) '.mat']));
                %electrodes_info{idx} = ncs_file_name.name;
                idx = idx+1;
            end
        
        case 'BlackRock'
            fprintf('Loading BlackRock file..\n');
            switch elec_type 
                case 'micro'
                    pattern = '*.ns5'; 
                case 'macro'
                    pattern = '*.ns3';
            end
            ns_files = dir(fullfile(base_folder, 'Raw', elec_type, pattern));
            ns_files
            assert(length(ns_files)==1, 'A SINGLE ns5 file in folder is expected')
            for ns_file_name=ns_files'
                file_name = ns_file_name.name;
                fprintf('Loading file %s\n', file_name)
                NS5=openNSx(fullfile(base_folder, 'Raw', elec_type, ns_file_name.name),'precision','double','uv') % precision should be set to 'double' ('short' is only for very large files)
                sr = NS5.MetaTags.SamplingFreq
                samplingInterval = 1/sr;
                timeend_sec = NS5.MetaTags.DataDurationSec
                idx=1;
                for elec = 1:size(NS5.Data, 1)
                   elec_name = NS5.ElectrodesInfo(elec).Label;
                   fprintf('CSC of channnel %d (%s)',elec, elec_name);
                   data = NS5.Data(elec, :);
                   fprintf('Sampling freq:')
                   %fprintf('%i\n', SampleFrequencies(1))
                   fprintf('%i\n', sr)
                   fprintf('%s\n', elec_name)
                   if startsWith(elec_name, 'elec')
                       save(fullfile(output_path,['CSC' extractAfter(elec_name, 4) '.mat']),'data','samplingInterval', 'elec_name', 'sr');
                   elseif startsWith(elec_name, 'ainp')
                       save(fullfile(output_path,['AINP' extractAfter(elec_name, 4) '.mat']),'data','samplingInterval', 'elec_name', 'sr');
                   elseif startsWith(elec_name, 'chan')
                       save(fullfile(output_path,['CSC' extractAfter(elec_name, 4) '.mat']),'data','samplingInterval', 'elec_name', 'sr');
                   end
                   %electrodes_info{idx} = elec_name;
                   idx = idx+1;
                end
                
            end
end
fclose(fid);
%save(fullfile(base_folder, 'electrodes_info_names.mat'), 'electrodes_info')
