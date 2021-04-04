function [] = sync_by_beeps(data)
% sync_by_beeps    

% Author: Ariel Tankus.
% Created: 08.08.2010.


sr = neuroport_samp_freq_hz;
notch_filter_half_width = 10;
freq_hz = 400;
freq_rel = freq_hz ./ sr;
[b, a] = ellip(2, 0.5, 20, [freq_hz-notch_filter_half_width, ...
                    freq_hz+ notch_filter_half_width]*2/sr);

data_filt = filtfilt(b, a, double(data));

% find envelope of abs signal:
w = rectwin(75);
c = conv(abs(data_filt), w);
len = length(c);

% threshold for suspected beeps:
t_mask = ((c > 1500) & (c < 2500));

% find initial points in each segment:
start_inds = find((t_mask(1:(end-1)) == 0) & (t_mask(2:end) == 1));

% if the points are less than 15000 indices (=0.5s) apart, consider them as 1:
d_small = find(diff(start_inds) < 15000);
% remove 2nd in the pairs of too-short distance:
start_inds(d_small + 1) = [];
save trigger_start_inds start_inds;

% NOTE: RUN beeps_verify_triggers.m and manually remove any trigger that is
% incorrect.

load trigger_start_inds_manual.mat;

d = diff(start_inds ./ sr);
suspected_inds = find(d < 2)+1;

export_beep_ttls_to_mouse_recording(start_inds);

plot_start_ind = 1.4E7;                % #indices.
plot_len = 4E6;                        % #indices.
s = start_inds((start_inds>=plot_start_ind) & ...
               (start_inds <= plot_start_ind+plot_len-1));
plot(t_mask(plot_start_ind:(plot_start_ind+plot_len-1)) .* ...
     c(plot_start_ind:(plot_start_ind+plot_len-1)));
hold on
plot(s-plot_start_ind, t_mask(s).*c(s), '.r');
