function [new_first_edge, new_last_edge, new_first_edge_ind, ...
          new_last_edge_ind, new_firing_rates, new_firing_rates_norm, ...
          ext_first_edge_ind, ext_last_edge_ind, ext_first_edge, ...
          ext_last_edge, ext_firing_rates_norm] = ...
    intersect_fr_with_interval(first_edge, last_edge, time_slot_length, ...
            interval, firing_rates_norm, firing_rates)
% intersect_fr_with_interval    Intersect the firing rates vector and
%                               normalized firing rates vector with a given
%                               time interval (typically, interval where
%                               kinematic recording took place).
%                               Also computes an extended normalized firing
%                               rates vector, which leaves extra firing rates
%                               at the beginning and end of the vector.
%                               This is particularly useful for correlation
%                               computations, since it allows to shift the
%                               firing rates vector without loosing events.
%                               [Add compact support to the kinematic vector
%                               in this case.]

% Author: Ariel Tankus.
% Created: 02.03.2006.


ext_len_sec = 3;      % length of time extension in sec.

new_first_edge = max(first_edge, ...
                     floor(interval(1) ./ time_slot_length).*time_slot_length);
new_last_edge  = min(last_edge, ...
                     ceil(interval(2) ./ time_slot_length).*time_slot_length);

if (new_last_edge <= new_first_edge)
    error(['new_last_edge (%g) <= new_first_edge (%g).  Check that ' ...
           'time_stamps_cache.mat is up-to-date and that electrode_info.mat '...
           'exists (for Neuroport sessions).'], new_last_edge, new_first_edge);
end

new_first_edge_ind = round((new_first_edge - first_edge) ./ time_slot_length)+1;
new_last_edge_ind  = round((new_last_edge - first_edge) ./ time_slot_length)+1;

% -1: #elements in vector is -1 smaller than #edges:
new_firing_rates = firing_rates(new_first_edge_ind:(new_last_edge_ind-1));
new_firing_rates_norm = ...
    firing_rates_norm(new_first_edge_ind:(new_last_edge_ind-1));

ext_len_bins = round(ext_len_sec ./ time_slot_length);  % #bins for extension.
ext_first_edge_ind = max(new_first_edge_ind - ext_len_bins, 1);
ext_last_edge_ind  = min(new_last_edge_ind + ext_len_bins, ...
            length(firing_rates_norm));
ext_first_edge = (ext_first_edge_ind-1) .* time_slot_length + first_edge;
ext_last_edge  = (ext_last_edge_ind-1) .* time_slot_length + first_edge;

% -1: #elements in vector is -1 smaller than #edges:
ext_firing_rates_norm = ...
    firing_rates_norm(ext_first_edge_ind:(ext_last_edge_ind-1));
