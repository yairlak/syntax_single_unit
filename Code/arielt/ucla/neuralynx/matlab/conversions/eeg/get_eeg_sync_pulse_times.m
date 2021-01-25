function [upstrokes, downstrokes, eeg_times] = ...
    get_eeg_sync_pulse_times(eeg_mat_file)

% Author: Ariel Tankus.
% Created: 14.09.2008.


load(eeg_mat_file);

x1_ch = find(all(header.channelname(:, 1:2) == repmat('X1', header.channels, 1), 2));
if (length(x1_ch) == 0)
    error('Cannot find sync pulse channel X1');
end
if (length(x1_ch) > 1)
    error('Ambiguous channel names (too many starting with X1)');
end

[upstrokes_inds, downstrokes_inds] = find_sync_pulse_peaks(data(x1_ch, :));
upstrokes   = (upstrokes_inds   - 1) ./ header.samplerate(x1_ch);
downstrokes = (downstrokes_inds - 1) ./ header.samplerate(x1_ch);

eeg_times = sort([upstrokes, downstrokes])';
