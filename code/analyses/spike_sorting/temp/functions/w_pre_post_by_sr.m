function [w_pre, w_post] = w_pre_post_by_sr(sr)
% w_pre_post_by_sr    Set the number of sample points that consists a spike
%                     waveform according to the sampling rate.
%
%                     [w_pre, w_post] = w_pre_post_by_sr(sr)
%                     sr            - 1x1 - sampling rate [Hz].
%                     w_pre, w_post - 1x1 - #sample points, before and after
%                                           the spike peak, that will compose
%                                           the spike waveform.
%
%                     See also:  Get_spikes_CSC, Get_spikes_CSC_notch2k_ariel,
%                                set_joint_parameters_CSC.

% Author: Ariel Tankus.
% Created: 23.02.2008.


% ARIEL: w_pre and w_post (#of pre- and post-spike observations considered as
% the spike waveform) depend on the sampling rate.
%if ((sr > 26000) & (sr < 29000))
%    w_pre_orig=20;               %number of pre-event data points stored
%    w_post_orig=44;              %number of post-event data points stored
%elseif ((sr > 30000) & (sr < 33000))
%    w_pre=22;               %number of pre-event data points stored
%    w_post=48;              %number of post-event data points stored
%elseif ((sr > 13000) & (sr < 15000))
%    w_pre=8;                %number of pre-event data points stored
%    w_post=24;              %number of post-event data points stored
%else
%    error(sprintf(['Unknown sampling rate (%dHz).  Set w_pre and w_post for '...
%                   'the new sampling rate in w_pre_post_by_sr().'], sr));
%end
w_pre_orig=20;               %number of pre-event data points stored
w_post_orig=44;              %number of post-event data points stored
orig_sr = 1E6 ./ 36;
w_pre  = round(w_pre_orig  / orig_sr .* sr);
w_post = round(w_post_orig / orig_sr .* sr);
