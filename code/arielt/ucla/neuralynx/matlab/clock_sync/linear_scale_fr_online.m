function [] = linear_scale_fr_online(b)
% linear_scale_mouse_recording    

% Author: Ariel Tankus.
% Created: 25.07.2005.


c = read_complete_fr_online('mouse_recording_relative.log');
num_lines = length(c{2});
ok_vec = false(num_lines, 1);

for i=1:num_lines
    
    [x, ok] = str2num(c{2}{i});
    
    if (ok)
        ok_vec(i) = true;
        % express Cheetah event times in Cheetah's clock:
        c{2}{i} = b(1).*c{2}{i}+ b(2);
        c{2}{i} = floor(c{2}{i});  % c{1} is in microseconds, so the fraction is
                                   % unnecessary.
        c{2}{i} = num2str(c{2}{i});
    end
end

if (exist('GapTimes.mat', 'file'))
    load GapTimes.mat;
    is_in_gap = is_in_time_intervals(c{1}', gap_times);
else
    is_in_gap = false(length(c{1}), 1);
end

fid = fopen('mouse_recording_in_cheetah_clock.log', 'w');
for i=1:length(c{2})
    if (~is_in_gap(i))
        fprintf(fid, '%.0f %s\n', c{1}(i), c{2}{i});
    else
        fprintf(fid, '%.0f Event: GAP_IN_RECORDING xxx %s xxx\n', c{1}(i), c{2}{i});
    end
end
fclose(fid);
