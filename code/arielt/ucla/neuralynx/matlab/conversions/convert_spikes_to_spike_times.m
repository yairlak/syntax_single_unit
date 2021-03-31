function [] = convert_spikes_to_spike_times()
% convert_spikes_to_spike_times    

% Author: Ariel Tankus.
% Created: 16.05.2005.


tic;
for i=1:get_num_channels
    fprintf('Converting %d ...', i);

    fname = sprintf('CSC%d_spikes.mat', i);
    
    if (exist(fname, 'file'))
        load(fname);

        if (exist('index_ts', 'var'))
            index = index_ts;
        end
        
        % save times only
        spike_times_sec = index ./ 1000;   % spike times in seconds.
        save(['CSC', num2str(i), '_spike_times'], 'spike_times_sec');

        fprintf('Done.\n');
        
        clear index;    % avoid propagation to next iteration.
    else
        fprintf('File does not exist, skipping!\n');
    end
end
toc
