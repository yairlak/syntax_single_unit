function [] = find_cheetah_neuroball_xcorr()
% find_cheetah_neuroball_xcorr    

% Author: Ariel Tankus.
% Created: 12.04.2005.


delta_thresh = 500;        % microseconds.

load cheetah_times_relative.txt
load neuroball_times_relative.txt

d_cheetah   = diff(cheetah_times_relative);
d_neuroball = diff(neuroball_times_relative);

c = xcorr(d_cheetah, d_neuroball);
figure
plot(c)

i = find(c == max(c))

if (length(d_cheetah) < length(d_neuroball))
    fprintf(['\nWarning: length(d_cheetah) < length(d_neuroball).\n\n', ...
             '         Please verify that the offset is taken in the ', ...
             'correct direction!\n\n']);
end

if (length(d_cheetah) >= length(d_neuroball))

    offset = i - length(d_cheetah);
    fprintf('Offset = %d\n', offset);

    inds = max(1+offset, 1):min(length(d_neuroball)+offset, length(d_cheetah));
    delta_diff = d_cheetah(inds) - d_neuroball(inds - offset);
    correlative_indices = find(abs(delta_diff) < delta_thresh);

    % +1: to compensate for the initial diff (which results in a vector shorter
    %     by 1 than the original data).
    interpolate_time_stamps(cheetah_times_relative, ...
                neuroball_times_relative, correlative_indices + offset + 1, ...
                correlative_indices + 1);

else
    
    offset = i - length(d_neuroball);
    fprintf('Offset = %d\n', offset);

    if (offset < 0)
        delta_diff = d_cheetah((1-offset):(length(d_cheetah)-offset)) - ...
            d_neuroball;
    else
        delta_diff = d_cheetah((1-offset):(length(d_cheetah)-offset)) - ...
            d_neuroball;
    end
        

    inds = max(1+offset, 1):min(length(d_cheetah)+offset, length(d_neuroball));
    delta_diff = d_neuroball(inds) - d_cheetah(inds - offset);
    correlative_indices = find(abs(delta_diff) < delta_thresh);

    % +1: to compensate for the initial diff (which results in a vector shorter
    %     by 1 than the original data).
    interpolate_time_stamps(cheetah_times_relative, ...
                neuroball_times_relative, correlative_indices + 1, ...
                correlative_indices + offset + 1);

end
