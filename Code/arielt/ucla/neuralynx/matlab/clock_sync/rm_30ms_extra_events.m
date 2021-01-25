function rm_30ms_extra_events(cur_bit)


% Author: Ariel Tankus.
% Created: 19.03.2018.


if (nargin < 1)
    cur_bit = 0;
end

cheetah_event_times_bit = load(sprintf('cheetah_event_times_bit%d.log', cur_bit));

d = diff(cheetah_event_times_bit);
d_30ms = find((d > 29000) & (d < 32000));

% remove the 30ms extra events by erasing the first timestamp in the 30ms
% pair (i.e., if t2-t1 = 30ms, erase t1):
cheetah_event_times_bit(d_30ms) = [];

d_new = diff(cheetah_event_times_bit);
fid = fopen('manual_cheetah_events.txt', 'w');
if (fid < 0)
    error('Cannot open file manual_cheetah_events.txt for write.');
end

fprintf(fid, '%d\n', round(cheetah_event_times_bit));

fclose(fid);
