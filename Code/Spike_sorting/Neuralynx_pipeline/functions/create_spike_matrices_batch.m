function create_spike_matrices_batch(dirToAnalyze)
cd(dirToAnalyze);
% matlabpool('open')
% load goodChannels.mat

time_before = 200;
time_after = 900;


% load electrode_info.mat % This is output from Ariel's  neuroport2mat_all3.3
% need to change this paramter (i = electrode numbers to be flexible)

% load the montage files of this specific subject 
% files=dir('Montage*.mat'); 
% load(files.name) % load the montage 
% vars=who('Montage*');
% try 
% Montage=eval(vars{1});
% catch 
%     fprintf('You do not have the montage file in this directory\n')
%     return 
% end


for i =  getRelevantChannelsWithTimesfiles(pwd)
    current_channel = i;
    current_file = sprintf('times_CSC%d.mat', current_channel  ); % used to be i in regular code with out good channels 
    if exist(current_file)
        cluster_class = load(current_file, 'cluster_class');
        cluster_class = cluster_class.cluster_class;
        NumberOfClusters = max(cluster_class(:, 1));
        if (NumberOfClusters==0)
            disp(['Channel ' num2str(i) ' has zero clusters'])
        else
            disp(num2str(NumberOfClusters))
        end
        for j = 1:NumberOfClusters
            current_cluster = j;
            electrodes_montage_array = i;%cell2mat({ElectrodesMontage{1,1}.ElectrodeID});
            electrodes_montage_labels = num2str(i);%{ElectrodesMontage{1,1}.Label};
            
            try
            current_region = electrodes_montage_labels %char(electrodes_montage_labels(electrodes_montage_array==current_channel)); % replace current channel with i for earlier version 
            catch ex
                ex
            end
            create_spike_matrices(current_channel, current_cluster, time_before, time_after, current_region);
        end;
    else
        continue
    end
end;

% matlabpool('close')