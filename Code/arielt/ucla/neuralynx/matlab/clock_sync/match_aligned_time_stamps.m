function [indices_cheetah, indices_neuroball, inds] = ...
    match_aligned_time_stamps(max_offset, d_cheetah, d_neuroball, ...
            d_neuroball_max_offset_mask, delta_thresh)
% match_aligned_time_stamps    

% Author: Ariel Tankus.
% Created: 09.10.2008.


if (max_offset > 0)
    indices_cheetah = (1+max_offset):min(length(d_neuroball) + max_offset, ...
                                         length(d_cheetah));
    indices_neuroball = 1:length(indices_cheetah);
else
    indices_neuroball = (1-max_offset):min(length(d_neuroball), ...
                                           length(d_cheetah) - max_offset);
    indices_cheetah = 1:length(indices_neuroball);
end

% limit the indices of neuroball to those that were in the maximal chunks:
[indices_neuroball, indices_neuroball_ind, offset_mask_ind] = ...
    intersect(indices_neuroball, find(d_neuroball_max_offset_mask));
indices_cheetah   = indices_cheetah(indices_neuroball_ind);

fprintf('offset = %d\n', max_offset);

delta_d = d_cheetah(indices_cheetah) - d_neuroball(indices_neuroball);
% The threshold (delta_thresh) is an estimation for about 1 sec. (i.e.,
% 1E6 microsecs).  The difference should be relative to the period of time
% during which the error is accumulated.
inds = find((abs(delta_d) ./ d_cheetah(indices_cheetah) < delta_thresh./1E6)&...
            (abs(d_cheetah(indices_cheetah) - ...
                 d_neuroball(indices_neuroball)) < delta_thresh));

fprintf('No. of matched indices: %d\n', length(inds));
if (isempty(inds))
    fprintf(['WARNING: No matching has been established between paradigm and Cheetah ' ...
           'clocks.\n\n']);
    return;
end

if (length(inds) ~= inds(end))
    % `Holes' appear in the matches: i.e., matched events are not consecutive.
    fprintf('\nWARNING: Suspected gaps in matched indices.\n\n');
end
