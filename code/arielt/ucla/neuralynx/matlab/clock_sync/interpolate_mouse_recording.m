function [ext_mouse_recording, edge_inds] = interpolate_mouse_recording(...
    mouse_recording, firing_rates_edges)
% interpolate_mouse_recording    Interpolate a mouse recording table to times
%                                between firing rate events (i.e., edges of
%                                the firing rate).
%
%                                [ext_mouse_recording, edge_inds] = ...
%                                    time_intervals_split(mouse_recording, ...
%                                        first_edge, time_slot_length, last_edge)
%                                mouse_recording - nx4 - [t, x, y, actor#] -
%                                        a table of mouse events.
%                                firing_rates_edges - 1xk - edges between
%                                        which the firing rates were computed.
%                                        This usually takes the form:
%                                        first_edge:time_slot_length:last_edge
%                                        and must be sorted [sec].
%
%                                See also: time_intervals_intersect,
%                                          convert_spike_times_to_fr,
%                                          convert_fr_to_fr_norm.

% Author: Ariel Tankus.
% Created: 14.06.2005.


min_diff = 1E-4;

% interpolate x-axis:
x_interp = interp1(mouse_recording(:, 1), mouse_recording(:, 2), ...
            firing_rates_edges);
% interpolate y-axis:
y_interp = interp1(mouse_recording(:, 1), mouse_recording(:, 3), ...
            firing_rates_edges);

non_nan_inds = find(~isnan(x_interp));
x_interp = x_interp(non_nan_inds);
y_interp = y_interp(non_nan_inds);
firing_rates_edges = firing_rates_edges(non_nan_inds);

% add the interpolated points to the mouse recording:
ext_mouse_recording_unsorted = [mouse_recording; firing_rates_edges', x_interp',...
                    y_interp', zeros(size(firing_rates_edges'))];
[ext_mouse_recording, ext_mouse_recording_inds] = ...
    sortrows(ext_mouse_recording_unsorted);
% we added the interpolated data at the end of the original mouse_recording
% array, so any index greater then size(mouse_recording, 1) belongs to the
% interpolated data.
edge_inds = find(ext_mouse_recording_inds > size(mouse_recording, 1));

% The interpolated data may be very close to an original mouse recording.
% This will result in a highly inaccurate calculation of velocity and
% acceleration, as the time interval is very short.
% We therefore eliminate such short intervals.

pos_inds = edge_inds(edge_inds > 1);
pos_d = ext_mouse_recording(pos_inds) - ext_mouse_recording(pos_inds - 1);
% remove the original pos_inds, and remain with the new pos_inds, which
% correspond to the firing rate edges:
rm_inds = pos_inds(pos_d < min_diff) - 1; 

neg_inds = edge_inds(edge_inds < length(edge_inds));
neg_d = ext_mouse_recording(neg_inds + 1) - ext_mouse_recording(neg_inds);
% remove the original neg_inds, and remain with the new neg_inds, which
% correspond to the firing rate edges:
rm_inds = [rm_inds; neg_inds(neg_d < min_diff) + 1];
rm_inds = sort(rm_inds);

rm_logical = logical(zeros(size(ext_mouse_recording, 1), 1));
rm_logical(rm_inds) = true;
rm_logical_cumsum = cumsum(rm_logical);     % for index offset

% actual removal:
ext_mouse_recording = ext_mouse_recording(~rm_logical, :); 
edge_inds = edge_inds - rm_logical_cumsum(edge_inds);
