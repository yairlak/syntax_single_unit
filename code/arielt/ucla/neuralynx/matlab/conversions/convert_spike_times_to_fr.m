function convert_spike_times_to_fr(time_slot_length, first_ch, last_ch, ...
            fname_suffix)
% convert_spike_times_to_fr    

% Author: Ariel Tankus.
% Created: 21.05.2005.

if (nargin < 1)
    first_ch = 1;
    last_ch  = get_num_channels;
end

if (nargin < 3)
    time_slot_length = 0.100;     % sec.
end

if (nargin < 4)
    fname_suffix = '_spike_times';
end

for i=first_ch:last_ch
    fprintf('Converting spike times to firing rates for channel %d ... ', i);

    fname = sprintf('CSC%d%s.mat', i, fname_suffix);
    if (exist(fname, 'file'))
        load(fname);
        if (strcmp(fname_suffix, '_cluster'))
            [rec_interval, rec_length] = get_rec_interval_cl(i);
        else
            [rec_interval, rec_length] = get_rec_interval(i);
        end

        [firing_rates, first_edge, last_edge] = ...
            spike_times_to_firing_rates(spike_times_sec, time_slot_length, rec_interval);
        save(sprintf('CSC%d_fr.mat', i), 'firing_rates', 'first_edge', 'last_edge', ...
             'time_slot_length');
        fprintf('Done.\n');
    else
        fprintf('File does not exist, skipping!\n');
    end
end
