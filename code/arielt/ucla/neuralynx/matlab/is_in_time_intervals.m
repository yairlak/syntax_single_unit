function [b, ind, spike_time_intervals_start, spike_time_intervals_end, multi_spike_times] = ...
    is_in_time_intervals(t, time_intervals)
% is_in_time_intervals    Is a given time inside a series of time intervals?
%
%                         [b, ind] = is_in_time_intervals(t, time_intervals)
%                         t              - 1xn - times.
%                         time_intervals - kx2 - [start, end] - time intervals.
%                         b              - 1xn - bool - true iff time t is
%                                                contained in any of the time
%                                                intervals.
%                         ind            - lx1 - (opt.) the index of the time
%                                                interval where t appears.
%                                                (empty if t is not contained
%                                                in any interval.)
%                                                If there are overlaps
%                                                between the intervals and t
%                                                appears in more than one,
%                                                the index of the first
%                                                interval only is provided.
%
%                         See also: get_time_intervals_inside_domain.

% Author: Ariel Tankus.
% Created: 15.03.2005.
% Modified: 31.03.2005.


if (isempty(time_intervals))
    b = false(size(t));
    if (nargout >= 2)
        ind = [];
        spike_time_intervals_start = [];
        spike_time_intervals_end = [];
        multi_spike_times = [];
    end
    return;
end

multi_t = repmat(t, size(time_intervals, 1), 1);
multi_time_intervals_start = repmat(time_intervals(:, 1), 1, length(t));
multi_time_intervals_end   = repmat(time_intervals(:, 2), 1, length(t));

mask_is_inside = ((multi_t >= multi_time_intervals_start) & ...
                  (multi_t <= multi_time_intervals_end));

b = any(mask_is_inside, 1);    % is t in any of the intervals?

if (nargout >= 2)
    % index of the time interval (the row index is the index of the time
    % interval where the corresponding spikes resides).
    % cumsum is used in order to get the first containing time interval at
    % every column.
    spike_time_intervals_start = multi_time_intervals_start(mask_is_inside);
    spike_time_intervals_end   = multi_time_intervals_end(mask_is_inside);
    multi_spike_times          = multi_t(mask_is_inside);

    % the (:) ensures a vertical vector; for a single time interval, logical
    % indices into the matrix (which is, in fact, a vector), produce a
    % horizontal vector.
    spike_time_intervals_start = spike_time_intervals_start(:);
    spike_time_intervals_end   = spike_time_intervals_end(:);
    multi_spike_times          = multi_spike_times(:);

    [ind, ind_col] = find(mask_is_inside & (cumsum(mask_is_inside) == 1));
    ind = ind';
end
