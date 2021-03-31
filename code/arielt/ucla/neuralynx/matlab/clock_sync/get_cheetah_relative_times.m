function relative_times = get_cheetah_relative_times(cheetah_bit)
% get_cheetah_relative_times    Get the relative time stamps of the sync
%                               pulses recorded by Cheetah from an
%                               Events.Nev file.
%
%                               relative_times = ...
%                                   get_cheetah_relative_times(cheetah_bit) 
%                               cheetah_bit      - 1x1 - {0,...,15} - Bit number
%                                                    to which TTLs are assigned.
%                                                    To use a manual file of
%                                                    Cheetah TTL times, use
%                                                    higher bits (>=16).
%
%                               See also: find_cheetah_neuroball_offset
%                                         cheetah_neuroball_time_offset_and_interp

% Author: Ariel Tankus.
% Created: 06.05.2005.
% Modified: 24.04.2009.  Added cheetah_bit.

if ((nargin >= 1) && (~isempty(cheetah_bit)))
    if (exist('manual_cheetah_events.txt', 'file'))
        cheetah_times = load('manual_cheetah_events.txt');
    else
        cheetah_times = load(sprintf('cheetah_event_times_bit%d.log', cheetah_bit));
    end
else
    % Default:  No manual events file, so load Events.Nev:
    [upstrokes, downstrokes] = load_nev('Events.Nev');
    cheetah_times = sort([upstrokes, downstrokes])';
end

if (strcmp(get_sys_type, 'Spike2'))
    % In Spike2 time is relative:
    relative_times = cheetah_times;
    return;
end

if (strcmp(get_sys_type, 'Alpha-Omega'))
    % In Alpha-Omega time is relative:
    relative_times = cheetah_times;
    return;
end

% We need Cheetah RELATIVE times, because wave_clus uses it for spike times:
[first_time_stamps, last_time_stamp]=read_first_last_time_stamp('CSC1.Ncs');

if (use_abs_cheetah_time)
    relative_times = cheetah_times;    % 2009-07-27: absolute Cheetah times.
else
    relative_times = cheetah_times - first_time_stamps(1);
end
