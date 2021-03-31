function linear_scale_mouse_recording(b, mouse_basename)
% linear_scale_mouse_recording    

% Author: Ariel Tankus.
% Created: 25.07.2005.


c = read_complete_mouse_recording([mouse_basename, '_relative.log']);

% express NeuroBall event times in Cheetah's clock:
c{1} = b(1).*c{1} + b(2);
c{1} = floor(c{1});    % c{1} is in microseconds, so the fraction is
                       % unnecessary.

if (exist('GapTimes.mat', 'file'))
    load GapTimes.mat;
    is_in_gap = is_in_time_intervals(c{1}', gap_times);
else
    is_in_gap = false(length(c{1}), 1);
end

fid = fopen([mouse_basename, '_in_cheetah_clock.log'], 'w');
for i=1:length(c{2})
    if (~is_in_gap(i))
        fprintf(fid, '%.0f %s\n', c{1}(i), c{2}{i});
    else
        fprintf(fid, '%.0f Event: GAP_IN_RECORDING xxx %s xxx\n', c{1}(i), c{2}{i});
    end
end
fclose(fid);
