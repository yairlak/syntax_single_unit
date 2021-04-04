function interval = get_recording_interval(force_unix)
% get_recording_interval    

% Author: Ariel Tankus.
% Created: 23.07.2005.


if (nargin == 0)
    force_unix = false;
end

if ((~force_unix) && (exist('mouse_level_all_intervals.log', 'file')))
    load mouse_level_all_intervals.log;
    if (~isempty(mouse_level_all_intervals))
        interval = ...
            [mouse_level_all_intervals(1), mouse_level_all_intervals(end)]./1E6;
    else
        [s, w] = unix('get_recording_interval');
        if (s > 0)
            error('Unable to get recording length.');
        end
        interval = str2num(w) ./ 1E6;
    end
else
    [s, w] = unix('get_recording_interval');
    if (s > 0)
        error('Unable to get recording length.');
    end
    load recording_interval.txt;
    interval = recording_interval ./ 1E6;
end
