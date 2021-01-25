function [firing_rates, first_edge, last_edge] = ...
    spike_times_to_firing_rates(spike_times_sec, time_slot_length, rec_interval)
% spike_times_to_firing_rates    Calculate the firing rates based on the times when
%                         spikes occured.  Non-overlapping time intervals are
%                         employed.
%
%                         firing_rates = spike_times_to_firing_rates(...
%                                                spike_times_sec, time_slot_length)
%                         spike_times_sec - 1xk - times when spikes were
%                                       detected in seconds.
%                         time_slot_length - 1x1 - length of the time slot [sec.]
%                                       in which rates are calculated.
%                                       Non-overlapping intervals of this
%                                       length are used.
%                         firing_rates - mx1 - each vector element specifies
%                                       the number of spikes during the
%                                       corresponding time interval.

% Author: Ariel Tankus.
% Created: 21.05.2005.


first_edge = floor(rec_interval(1) ./ time_slot_length).*time_slot_length;
last_edge  = ceil(rec_interval(2)  ./ time_slot_length).*time_slot_length;
edges = first_edge:time_slot_length:last_edge;

if (isempty(spike_times_sec))
    firing_rates = zeros(round((last_edge - first_edge) ./ time_slot_length), 1);
    return;
end

firing_rates = histc(spike_times_sec, edges)';

% Join the last bin to its predecessor.
% histc counts the number of elements exactly equal to the last edge,
% and put them in the last bin.  Thus the last firing rate includes the
% case of spikes at the time of its upper edge (unlike all other bins,
% which exclude their upper bin).
firing_rates(end-1) = firing_rates(end-1) + firing_rates(end);
firing_rates = firing_rates(1:(end-1));
