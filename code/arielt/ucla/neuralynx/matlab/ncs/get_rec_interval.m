function [rec_interval, rec_length] = get_rec_interval(ch)
% get_rec_interval    

% Author: Ariel Tankus.
% Created: 17.02.2010.


fname = sprintf('.%stimes_CSC%d.mat', filesep, ch);
if (exist(fname, 'file'))
    load(fname, 'time0', 'timeend');
end

% compute recording interval and duration:
if (exist('time0', 'var'))
    if (length(time0) == 1)
        rec_interval = [time0, timeend] ./ 1E6;    % microsec.-->sec.
    else
        rec_interval = [time0(ch), timeend(ch)] ./ 1E6;    % microsec.-->sec.
    end
else
    [first_time_stamps, last_time_stamp] = ...
        read_first_last_time_stamp(sprintf('CSC%d.Ncs', ch));
    rec_interval = [first_time_stamps(1), last_time_stamp] ./ 1E6; % microsec.-->sec.
end
rec_length = diff(rec_interval);
if (~use_abs_cheetah_time)
    rec_interval = [0, rec_length];
end
