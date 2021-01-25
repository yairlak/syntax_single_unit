function [cheetah_times_relative, paradigm_times_relative] = ...
    load_input_signals(is_eeg, mouse_basename, cheetah_bit)
% load_input_signals    Load the input signals for time synchronization.
%                       By default, loads the sync pulse times of Cheetah
%                       and mouse recording.  If is_eeg==true, replaces
%                       the time stamps of the mouse recording by that of
%                       the Stellate EEG recording.
%
%                       [cheetah_times_relative, mouse_recording_times_relative] = ...
%                           load_input_signals(is_eeg, mouse_basename)
%                       is_eeg - 1x1 - logical - true iff synchronizing
%                                                EEG instead of mouse recording.
%                       mouse_basename - 1x1 - string - basename of the events
%                                                log file to process.
%                       cheetah_bit      - 1x1 - {0,...,15} - Bit number
%                                                to which TTLs are assigned.
%                       cheetah_times_relative - nx1 - time stamps of the sync 
%                                                pulses recorded on Cheetah.
%                       paradigm_times_relative - nx1 - time stamps of the sync
%                                                pulses from the mouse
%                                                recording or EEG.
%
%                       See also: get_cheetah_relative_times,
%                                 get_eeg_relative_times,
%                                 cheetah_neuroball_time_offset_and_interp.

% Author: Ariel Tankus.
% Created: 09.10.2008.
% Modified: 24.04.2009.  Added mouse_basename.


cheetah_times_relative = get_cheetah_relative_times(cheetah_bit);
if (~is_eeg)
    paradigm_times_relative = load([mouse_basename, '_times_relative.txt']);
else
    % put EEG times in mouse_recording_times_relative (to sync to Cheetah time):
    paradigm_times_relative = get_eeg_relative_times(mouse_basename);
end

if (exist(['.', filesep, 'ttl_filter_odd'], 'file'))
    % only odd paradigm TTLs were recorded:
    paradigm_times_relative = paradigm_times_relative(1:2:end);
elseif (exist(['.', filesep, 'ttl_filter_even'], 'file'))
    paradigm_times_relative = paradigm_times_relative(2:2:end);
end

if (isempty(paradigm_times_relative))
    error('Empty %s relatives times.  Are you in the right dir?', mouse_basename);
end
