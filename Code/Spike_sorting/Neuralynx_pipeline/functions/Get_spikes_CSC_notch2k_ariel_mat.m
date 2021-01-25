function Get_spikes_CSC_notch2k_ariel_mat(channels, data_folder, not_neuroport, samp_freq_hz)
% function Get_spikes_CSC(channel,TimeStamps)
% Gets spikes from all channels in the .mat file channels. This batch is
% to be used with Neuralynx data.
% Saves spikes and spike times.

% Modified: 23.07.2009 (Ariel, Roy & Omri).  Order of ellip filter
%                      changed from 2 to 4.
fid = fopen('1failedChannels.txt','w+');
if nargin <1
    load channels
end

%    % make gap_inds relative to filled-in time stamps:
%    gap_inds(2:end) = gap_inds(2:end) + cumsum(gap_lens);

handles.par.detection = 'pos';              %type of threshold
handles.par.stdmin = 5;                     %minimum threshold
handles.par.stdmax = 50;                    %maximum threshold
handles.par.interpolation = 'y';            %interpolation for alignment
handles.par.int_factor = 2;                 %interpolation factor
handles.par.detect_fmin = 300;              %high pass filter for detection - previously 300
handles.par.detect_fmax = 3000;             %low pass filter for detection - previously 1000
handles.par.sort_fmin = 300;                %high pass filter for sorting
handles.par.sort_fmax = 3000;               %low pass filter for sorting
handles.par.segments_length = 5;            %length of segments in 5' in which the data is cutted.
notch_filter_half_width = 1;                % Hz.
high_voltage_thresh = 750;                  % No spikes with voltage above
% this threshold.
if ((nargin < 3) || (~not_neuroport))
    % Neuroport only:
    scale_factor = 250/1000;   % 250nV per LSB
else
    scale_factor = 1;
end

%load('D:\Experiments\Patient011\Data\ChannelsCSC\electrode_info.mat');

for k=1:length(channels)
    try
        tic
        
        channel = channels(k)
        
        load(fullfile(data_folder, ['CSC' num2str(channel) '.mat']));
        samp_interval_micsec = samplingInterval.*1000;  % same as Cheetah's.
        TimeStamps = 0:samp_interval_micsec:((length(data)-1).*samp_interval_micsec);
        
        %Find the starting of the recording and gets sampling frequency
        time0 = TimeStamps(1);
        timeend = TimeStamps(end);
        sr = samp_freq_hz;      % ARIEL: sampling rate in Hz.
        
        % ARIEL: sr/1000 = #samples in 1ms.
        % ARIEL: #samples in refrectory period (`ref') is taken as 1.5 this #.
        %        (i.e., #samples in 1.5ms).
        ref = floor(1.5 *sr/1000);       %minimum refractory period (in ms)
        handles.par.ref = ref;
        handles.par.sr = sr;
        [handles.par.w_pre, handles.par.w_post] = w_pre_post_by_sr(sr);
        
        lts = length(TimeStamps);
        handles.par.segments = ceil((timeend - time0) / ...
            (handles.par.segments_length * 1e6 * 60));
        
        %That's for cutting the data into pieces
        segmentLength = floor (length(TimeStamps)/handles.par.segments);
        tsmin_ind = 1 : segmentLength :length(TimeStamps);
        tsmin_ind = tsmin_ind(1: handles.par.segments);
        tsmax_ind = tsmin_ind - 1;
        tsmax_ind = tsmax_ind(2:end);
        tsmax_ind = [tsmax_ind, length(TimeStamps)];
        recmax=tsmax_ind;
        recmin=tsmin_ind;
        tsmin = TimeStamps(int64(tsmin_ind));
        tsmax = TimeStamps(int64(tsmax_ind));
        %clear TimeStamps;
        
        index_all  = [];
        spikes_all = [];
        
        BadFrequenciesAccum          = cell(1, length(tsmin));
        BadFrequenciesThresholdAccum = NaN(1, length(tsmin));
        
        HighVoltageIndsAll  = [];
        HighVoltageTimesAll = [];
        
        beginning_of_recordings_filtered = [];
        for j=1:length(tsmin)
            
            % LOAD CSC DATA
            if (length(data) < recmax(j))
                keyboard
            end
            x = -double(data(recmin(j):recmax(j))).*scale_factor;
            
            % detect and filter out bad frequencies:
            [BadFrequencies, BadFrequenciesThreshold] = GetBadFrequencies(x, sr);
            BadFrequenciesAccum{j} = BadFrequencies;
            BadFrequenciesThresholdAccum(j) = BadFrequenciesThreshold;
            for bf_index = 1:length(BadFrequencies)
                if ((bf_index == 1) || (BadFrequencies(bf_index) - ...
                        BadFrequencies(bf_index-1) >= notch_filter_half_width))
                    fprintf(['Channel %d, chunk %d/%d: Filtering frequency ', ...
                        '%.2f .\n'], channel, j, length(tsmin), ...
                        BadFrequencies(bf_index));
                    [b,a]=ellip(2,0.5,20,[BadFrequencies(bf_index)-notch_filter_half_width, BadFrequencies(bf_index)+notch_filter_half_width]*2/sr,'stop');
                    x=filtfilt(b,a,x);
                else
                    % No point in a separate filter for the current frequency, as
                    % it has already been filter due to the previous one, which
                    % was close enough to the current:
                    fprintf(['Channel %d, chunk %d/%d: Skipping frequency ', ...
                        '%.2f .\n'], channel, j, length(tsmin), ...
                        BadFrequencies(bf_index));
                end
                
            end
            HighVoltageInds     = find(x > high_voltage_thresh);
            HighVoltageIndsAll  = [HighVoltageInds, HighVoltageIndsAll + ...
                tsmin_ind(j) - 1];
            HighVoltageTimes    = (HighVoltageInds-1)*1e6/sr+tsmin(j);
            HighVoltageTimesAll = [HighVoltageTimesAll, HighVoltageTimes];
            
            % SPIKE DETECTION WITH AMPLITUDE THRESHOLDING
            handles.par.data_folder = data_folder;
            handles.par.channel = channel;
            handles.par.j = j;
            [spikes,thr,index] = amp_detect(x,handles.par);       %detection with amp. thresh.
            
            % ARIEL: was: index=index*1e6/sr+tsmin(j);
            % The above is a bug, which shifts all spike times by 1 sample (i.e.,
            % by 36 microsec.):  The index begins from 1, so the if the spike
            % occurred on the first time stamp, it would be assigned the time
            % stamp of: 1*1e6/27777+tsmin(j) = 36+tsmin(j), instead of tsmin(j):
            %       ARIEL: Comment out, 24.07.2009:
            %        index=(index-1)*1e6/sr+tsmin(j);
            index = index + recmin(j) - 1;
            index_all = [index_all index];
            spikes_all = [spikes_all; spikes];
        end
        
        %    index = (index_all-time0)/1000;
        %    HighVoltageTimesAll = (HighVoltageTimesAll - time0) ./ 1000;
        index = TimeStamps(index_all) ./ 1000;
        HighVoltageTimesAll = HighVoltageTimesAll ./ 1000;
        spikes = spikes_all;
        save(fullfile(data_folder, ['CSC', num2str(channel), '_spikes']), 'spikes', 'index', 'time0', ...
            'timeend');
        save(fullfile(data_folder, ['BadFrequencies' num2str(channel) '.mat']), 'BadFrequenciesAccum', ...
            'BadFrequenciesThresholdAccum');
        save(fullfile(data_folder, ['HighVoltageTimes' num2str(channel) '.mat']), 'HighVoltageTimesAll', ...
            'HighVoltageIndsAll');
        fprintf('Detected %d cases of too-high voltage.\n', ...
            length(HighVoltageTimesAll));
        toc
        clear('data', 'TimeStamps');
    catch ME
        fprintf(fid,'%d - Failed Channel \t %s\n', channels(k),ME.message);
    end
end
fclose(fid);
