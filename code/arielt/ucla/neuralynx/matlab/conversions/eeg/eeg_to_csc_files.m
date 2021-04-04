function eeg_to_csc_files(channels, wd, eeg_fname, b);

if (nargin >= 2)
    cur_wd = pwd;
    cd(wd);
end
if (nargin < 3)
    eeg_fname = 'eeg.mat';
end

t = tic;

load(eeg_fname);

cfg = header;

for k=1:length(channels)
    ch = channels(k);
    fprintf('Converting EEG channel %d\n', ch);
    eeg = data(ch, :);

    sr = header.samplerate(ch);
    time_slot_length = 1./sr;

    time_first_sec = 0;
    time_last_sec = (length(eeg)-1) .* time_slot_length;
    times_vec = 0:time_slot_length:time_last_sec;
    
    % convert to Cheetah times:
    time_first_sec_actual   = floor(b(1).*1E6.*time_first_sec + b(2)) ./ 1E6;
    time_last_sec_actual    = floor(b(1).*1E6.*time_last_sec  + b(2)) ./ 1E6;
    time_slot_length_actual = (time_last_sec_actual-time_first_sec_actual) ./...
        (length(eeg)-1);
    times_vec_actual = ...
        time_first_sec_actual:time_slot_length_actual:time_last_sec_actual;
    % interpolate to round time_slot_length bins:
    time_first_sec = ceil(time_first_sec_actual ./ time_slot_length).* ...
        time_slot_length;
    time_last_sec  = floor(time_last_sec_actual ./ time_slot_length).* ...
        time_slot_length;
    
    times_vec = time_first_sec:time_slot_length:time_last_sec; 
    eeg = interp1(times_vec_actual, double(eeg), times_vec);
    
    low_sr_time_interval  = time_slot_length;
    high_sr_time_interval = time_slot_length;
    
    first_edge    = time_first_sec - time_slot_length./2;
    last_edge     = time_last_sec  + time_slot_length./2;
   
    save(['CSC', num2str(ch), '_eeg'], 'eeg', 'time_first_sec', ...
         'time_last_sec', 'sr', 'low_sr_time_interval', ...
         'high_sr_time_interval', 'time_slot_length', 'first_edge', ...
         'last_edge', 'cfg');

end

if (nargin >= 2)
    cd(cur_wd);
end

toc(t);
