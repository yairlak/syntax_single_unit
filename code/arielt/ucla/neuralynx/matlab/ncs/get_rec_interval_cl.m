function [rec_interval, rec_length] = get_rec_interval_cl(cl)
% get_rec_interval_cl    Get recording length of a cluster.

% Author: Ariel Tankus.
% Created: 17.02.2010.


[ch, sub_cl] = cl_to_ch_sub_cl(cl);
[rec_interval, rec_length] = get_rec_interval(ch);
