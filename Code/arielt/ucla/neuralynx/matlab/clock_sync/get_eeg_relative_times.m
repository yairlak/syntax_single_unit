function relative_times = get_eeg_relative_times(mouse_basename)
% get_eeg_relative_times        Get the relative time stamps of the sync
%                               pulses recorded by the Stellate machine
%                               (EEG), based on a sync-pulses dedicated
%                               channel (usually X1).
%
%                               See also: find_cheetah_neuroball_offset
%                                         cheetah_neuroball_time_offset_and_interp
%                                         get_cheetah_relative_times.

% Author: Ariel Tankus.
% Created: 09.10.2008.


if (nargin < 1)
    mouse_basename = 'eeg';
end

[upstrokes, downstrokes, relative_times] = ...
    get_eeg_sync_pulse_times([mouse_basename, '.mat']);
relative_times = 1E6.*relative_times;   % convert to microsec.
