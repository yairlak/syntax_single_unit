function firing_rates_norm = normalize_fr(firing_rates, win_length)
% normalize_fr    Normalize a vector of firing rates.  That is, reduce the
%                 average from the firing rates vector and divide by the
%                 standard deviation.  The average and standard deviation are
%                 computed inside a sliding window of a given length.  The
%                 window precedes the current firing rate.
%
%                 firing_rates_norm = normalize_fr(firing_rates, win_length)
%                 firing_rates - nx1 - vector of firing rates.
%                 win_length   - 1x1 - length (#indices) of the window where
%                                      the average and standard deviations
%                                      are computed.
%                 firing_rates_norm - (n-win_length)x1 - normlized firing rates.
%
%                 See also: sliding_avg_std, convert_fr_to_fr_norm,
%                           convert_spike_times_to_fr.

% Author: Ariel Tankus.
% Created: 28.05.2005.


%[avg_fr, std_fr] = sliding_avg_std_sparse(sparse(firing_rates), win_length);
[avg_fr, std_fr] = sliding_avg_std(firing_rates, win_length);

% TESTING CODE:
%if (all(avg_fr == avg_fr_sparse))
%    fprintf('Sparse avg fr == avg fr\n');
%else
%    keyboard
%end
%if (all(abs(std_fr - std_fr_sparse) < 1E-10))
%    fprintf('Sparse std fr == std fr\n');
%else
%    keyboard
%end

% each normalized f.r. is computed based on the window immediately preceding
% the current f.r.
firing_rates_norm = firing_rates((win_length+1):end) - avg_fr(1:(end-1));
non_zero_std = (std_fr(1:(end-1)) ~= 0);

if (any(non_zero_std))
    firing_rates_norm(non_zero_std) = firing_rates_norm(non_zero_std) ./ ...
        std_fr(non_zero_std);
end
