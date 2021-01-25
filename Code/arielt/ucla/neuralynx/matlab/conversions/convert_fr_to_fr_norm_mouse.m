function [] = convert_fr_to_fr_norm_mouse(first_ch, last_ch, use_norm_fr)
% normalize_fr    

% Author: Ariel Tankus.
% Created: 28.05.2005.


if (nargin < 1)
    first_ch = 1;
    last_ch  = get_num_channels;
end

if (nargin < 3)
    use_norm_fr = true;
end

params = struct(...
    'normalize',      use_norm_fr, ...
    'smooth',         false, ...
    'win_len_sec',    60, ...
    'gauss_1sig_ms',  500/3 ...
);

save normalization_params params;

%gauss_win_length = 5;

t = cputime;

interval = get_recording_interval(true);

% make all normalized firing rates start from the same time and end at the
% same time:
for i=first_ch:last_ch
    fprintf('Normalizing channel %d ... ', i);
    fname = sprintf('CSC%d_fr.mat', i);
    if (exist(fname, 'file'))
        
        load(fname);

        firing_rates_norm = firing_rates ./ time_slot_length;  % convert to Hz.

        tic;
            win_length = round(params.win_len_sec/time_slot_length);
            %    [firing_rates_norm, std_fr] = sliding_avg_std(firing_rates_norm, win_length);
            if (params.normalize)
                if (params.smooth)
                    firing_rates_norm = smooth_start_stop_table(firing_rates_norm', ...
                                time_slot_length, params.gauss_1sig_ms)';
                    fprintf('\n\nSmoothing firing rates!!!\n\n');
                end
                firing_rates_norm = normalize_fr(firing_rates_norm, win_length);
                %    firing_rates_norm = normalize_fr_median(firing_rates_norm, win_length);
                first_edge = first_edge + win_length.*time_slot_length;
            end
        toc;
        
        %    firing_rates_norm = diff(firing_rates_norm);
        %    first_edge = first_edge + 1;
        
        %    win_length = 10;
        %    firing_rates_norm = conv(firing_rates_norm, rectwin(win_length));
        %    firing_rates_norm = firing_rates_norm(win_length:length(firing_rates));
        %    first_edge = first_edge + (win_length - 1).*time_slot_length;
        
        %        fr_len = length(firing_rates_norm);
        %
        %        firing_rates_norm = conv(firing_rates_norm, gausswin(gauss_win_length));
        %        firing_rates_norm = firing_rates_norm(ceil(gauss_win_length ./ 2):...
        %                                              (end-floor(gauss_win_length ./ 2)));
        %        if (length(firing_rates_norm) ~= fr_len)
        %            error(sprintf('Different lengths (%d ~= %d).', fr_len, ...
        %                        length(firing_rates_norm)));
        %        end
        %        fprintf('\nAVERAGING FIRING RATE!!!\n\n');
        [first_edge, last_edge, first_edge_ind, last_edge_ind, firing_rates, ...
         firing_rates_norm, ext_first_edge_ind, ext_last_edge_ind, ...
         ext_first_edge, ext_last_edge, ext_firing_rates_norm] = ...
            intersect_fr_with_interval(first_edge, last_edge, time_slot_length, ...
                    interval, firing_rates_norm, firing_rates);
        
        save(sprintf('CSC%d_fr_norm.mat', i), 'firing_rates_norm', ...
             'firing_rates', 'ext_first_edge', 'ext_last_edge', ...
             'ext_firing_rates_norm', 'first_edge', 'last_edge', ...
             'time_slot_length');
        
        fprintf('Done.\n');
    else
        fprintf('File does not exist, skipping!\n');
    end
end

print_cpu_time(t, last_ch - first_ch + 1);
