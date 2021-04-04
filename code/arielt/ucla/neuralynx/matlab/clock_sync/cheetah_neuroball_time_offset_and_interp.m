function cheetah_neuroball_time_offset_and_interp(is_eeg, mouse_basename, ...
            cheetah_bit)
% cheetah_neuroball_time_offset_and_interp    

% Author: Ariel Tankus.
% Created: 13.04.2005.
% Modified: 24.04.2009.  Added mouse_basename.
% Modified: 10.01.2017.  Bug fix: code never arrived at odd/even timestamp
%                        comparison even in case of failure in full matching.


thresh_small_spaced_microsec = 50000;     %=50ms

if (is_eeg)
    if ((~exist('channels.mat', 'file')) & (~exist('channels.txt', 'file')))
        error(['Please mark EEG channels to convert in file: ' ...
               'channels.mat']);
    end
    delta_thresh = 3000;   % EEG accuracy is 1/200Hz = 5ms, so matches have
                           % lower quality.
else
    delta_thresh = 1000;
end

[cheetah_times_relative, paradigm_times_relative] = ...
    load_input_signals(is_eeg, mouse_basename, cheetah_bit);

if (isempty(cheetah_times_relative))
    error('cheetah_times_relative is empty!');
end
if (isempty(paradigm_times_relative))
    error('paradigm_times_relative is empty!');
end

paradigm_times_relative_orig = paradigm_times_relative;

b = [0, 0];
for i=1:5
    
    [max_offset, d_cheetah, d_neuroball, cheetah_times_relative, ...
     paradigm_times_relative, d_neuroball_max_offset_mask] = ...
        find_cheetah_neuroball_offset(cheetah_times_relative, ...
                                      paradigm_times_relative, delta_thresh);

    [indices_cheetah, indices_neuroball, inds] = ...
        match_aligned_time_stamps(max_offset, d_cheetah, d_neuroball, ...
                                  d_neuroball_max_offset_mask, delta_thresh);

    if (~isempty(inds))
        % +1: to compensate for the removed index during `diff'.
        [b, b_w_scaling] = regress_time_stamps(cheetah_times_relative, ...
                                               paradigm_times_relative, ...
                                               indices_cheetah(inds) + 1, ...
                                               indices_neuroball(inds) + 1);

        if ((b_w_scaling(1) >= 0.99) && (b_w_scaling(1) <= 1.001))
            break;
        end
    else
        fprintf('\nSynchronization Failed!\n\n');
%        return;
    end

    if (i == 1)
        % in the next iteration, examine only odd TTLs:
        paradigm_times_relative = paradigm_times_relative_orig(1:2:end);
        fprintf('Checking odd TTLs...\n');
    elseif (i == 2)
        % in the next iteration, examine only even TTLs:
        paradigm_times_relative = paradigm_times_relative_orig(2:2:end);
        fprintf('Checking even TTLs...\n');
    elseif (i == 3)
        fprintf('Excluding small-spaced (<50ms) TTLs...\n');
        d = find(diff(cheetah_times_relative) < thresh_small_spaced_microsec);
        cheetah_times_relative(d+1) = [];   % +1: remove the 2nd event in the pair.
        %keyboard
    elseif (i == 4)
        fprintf('Excluding small-spaced (<50ms) TTLs...\n');
        d = find(diff(cheetah_times_relative) < thresh_small_spaced_microsec);
        cheetah_times_relative(d) = [];   % remove the 1st event in the pair.
        %keyboard
    else
        % fourth iteration, all attempts failed:
        if (isempty(inds))
            error(sprintf('No matching found.'));
        else
            error(sprintf('Extreme time scale: a = %g (b = %g).', b_w_scaling));
        end
    end

end

apply_linear_time_scaling(b, is_eeg, mouse_basename);

save([mouse_basename, '_time_scale_coeffs.mat'], 'b', 'b_w_scaling');
