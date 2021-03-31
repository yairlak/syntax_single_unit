function [upstrokes_inds, downstrokes_inds] = find_sync_pulse_peaks(x1_data)

% Author: Ariel Tankus.
% Created: 14.09.2008.


%volt_inc_th = 1500;    % Modified: 26.05.2010.
volt_inc_th = 2000;

d = diff(x1_data);

% find upstrokes:
inc_inds = find(d > volt_inc_th);

dd = diff(inc_inds);
dd = [-1000, dd];             % dummy, to ensure the first is included.
non_consec_up = find(dd ~= 1);   % find non-consecutive inds.

upstrokes_inds = inc_inds(non_consec_up);

% find downstrokes:
dec_inds = find(d < -volt_inc_th);

ddd = diff(dec_inds);
ddd = [-1000, ddd];             % dummy, to ensure the first is included.
non_consec_down = find(ddd ~= 1);   % find non-consecutive inds.

downstrokes_inds = dec_inds(non_consec_down);
