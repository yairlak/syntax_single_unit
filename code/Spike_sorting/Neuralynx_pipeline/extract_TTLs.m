clear; close all; clc;

%%
patient = 'patient_530';
recording_system = 'BlackRock';% BlackRock, Neuralynx

base_folder = ['/home/yl254115/Projects/syntax_single_unit/Data/UCLA/', patient];
% base_folder = ['/neurospin/unicog/protocols/intracranial/single_unit/Data/UCLA/', patient];
output_path = fullfile(base_folder,'ChannelsCSC');

% mkdir(output_path);
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


path2logs = '/home/yl254115/Projects/single_unit_syntax/Data/UCLA/patient_480/Logs/Raw_logs';

%%
switch recording_system
        case 'Neuralynx'
            ncs_files = dir([base_folder '/Raw/*.ncs']);
            idx=1;
            for ncs_file_name=ncs_files'
                file_name = ncs_file_name.name;
                ncs_file = fullfile(base_folder,'Raw',file_name);
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
            nev_file = dir([base_folder '/Raw/nev_files/*.nev']);
            nev_file = fullfile(base_folder,'Raw','nev_files', nev_file(1).name);
            NEV = openNEV(nev_file, 'read');
%                      
end

%%
TTLs = NEV.Data.SerialDigitalIO.TimeStampSec;
TTLs = double(TTLs)
% First block
IX = TTLs < 600; %~2e7 in samples
TTLs_1 = TTLs(IX);

%% Load logs
all_triggers = [];
files = dir(fullfile(path2logs, '*.log'));
for f = 1:length(files)
    FN = files(f).name;
    curr_f = fopen(fullfile(path2logs, FN));
    triggers = [];
    while ~feof(curr_f)
        curr_line = fgetl(curr_f);
        fields = split(curr_line);
        if strcmp(fields{2}, 'CHEETAH_SIGNAL')
            triggers = [triggers str2double(fields{4}) str2double(fields{1})];
        end
    end
    all_triggers{f} = triggers;
end

%%
% scatterall_triggers{1}, 1, '*')
% plot(TTLs_1, 1, '*')
block = 3;
curr_block_triggers = all_triggers{block};%(5:end);
diff_curr_block_triggers = diff(curr_block_triggers);
num_triggers = length(curr_block_triggers);
l = [];
for t = 1:(length(TTLs)-num_triggers)
    curr_TTLs = TTLs(t:(t+num_triggers-1));
    diff_curr_TTLs = diff(curr_TTLs);
    curr_l = norm(diff_curr_block_triggers-diff_curr_TTLs);
    l = [l, curr_l];
end

plot(l)
