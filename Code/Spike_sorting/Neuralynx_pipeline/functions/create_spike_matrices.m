function create_spike_matrices(channel_number, cluster_number, time_before, time_after, current_region)

    fn = sprintf('Channel%d_Cluster%d_%s', channel_number, cluster_number, current_region);
    
    % a structure of matrices with the stimulus onsets in microseconds the form of
    % StimulusTimes.stim1 = [ 234 ; 235 ; 236] ... StimulusTimes.stim2 = [345 , 356 , 378]....
    % load the stim files of this specific subject, in this specific run 
    files=dir('stimArrays*.mat');
    load(files.name) % load the montage
  %  vars=who('StimulusTimes');
    try
  %      stimuliArrayNew=eval(vars{1});
    catch
        disp('You do not have the the stim array .mat file in this directory')
        return
    end
    
    
    % convert these 30 hz times from BlackRock into MicroSeconds
    fieldNames=fieldnames(stimuliArrayNew); % this is the stim times 
    
    for i = 1:length(fieldNames)
        StimulusTimes.(fieldNames{i})=round(stimuliArrayNew.(fieldNames{i}) *(1e6/3e4)); %convert to micro secs from data at 30Khz
    end
    
    % loops on all conditions in StimulusTimes, feeding this function one
    % group of stim onsets at a time. 
    for i  = 1:length(fieldNames)
        BlockSpikeTrains.(fieldNames{i}) = GetBlockSpikeTrains_new(channel_number, cluster_number, StimulusTimes.(fieldNames{i}), time_before, time_after);
    end
    save([fn '.mat'],'BlockSpikeTrains');
    
%     BottleSpikes = GetBlockSpikeTrains_new(channel_number, cluster_number, StimulusTimes.Bottle, time_before, time_after);
%     ButtonSpikes = GetBlockSpikeTrains_new(channel_number, cluster_number, StimulusTimes.Button, time_before, time_after);
%     DoorKnobSpikes = GetBlockSpikeTrains_new(channel_number, cluster_number, StimulusTimes.DoorKnob, time_before, time_after);
%     DrawerSpikes = GetBlockSpikeTrains_new(channel_number, cluster_number, StimulusTimes.Drawer, time_before, time_after);
%     JoyStickSpikes = GetBlockSpikeTrains_new(channel_number, cluster_number, StimulusTimes.JoyStick, time_before, time_after);
%     KeyLockSpikes = GetBlockSpikeTrains_new(channel_number, cluster_number, StimulusTimes.KeyLock, time_before, time_after);
%     KeyPressSpikes = GetBlockSpikeTrains_new(channel_number, cluster_number, StimulusTimes.KeyPress, time_before, time_after);
%     OpenLidSpikes = GetBlockSpikeTrains_new(channel_number, cluster_number, StimulusTimes.OpenLid, time_before, time_after);
%     PenGraspSpikes = GetBlockSpikeTrains_new(channel_number, cluster_number, StimulusTimes.PenGrasp, time_before, time_after);
%     ScisorsSpikes = GetBlockSpikeTrains_new(channel_number, cluster_number, StimulusTimes.Scisors, time_before, time_after);
%     ShoeLaceSpikes = GetBlockSpikeTrains_new(channel_number, cluster_number, StimulusTimes.ShoeLace, time_before, time_after);
%     SodaCanSpikes = GetBlockSpikeTrains_new(channel_number, cluster_number, StimulusTimes.SodaCan, time_before, time_after);
%     XSpikes = GetBlockSpikeTrains_new(channel_number, cluster_number, StimulusTimes.X, time_before, time_after);
%     ZipperSpikes = GetBlockSpikeTrains_new(channel_number, cluster_number, StimulusTimes.Zipper, time_before, time_after);
% 
%     
%     save([fn '.mat'], 'BottleSpikes', 'ButtonSpikes', 'DoorKnobSpikes', 'DrawerSpikes', ...
%         'JoyStickSpikes', 'KeyLockSpikes', 'KeyPressSpikes', 'OpenLidSpikes', 'PenGraspSpikes', ...
%         'ScisorsSpikes', 'ShoeLaceSpikes', 'SodaCanSpikes', 'XSpikes', 'ZipperSpikes');