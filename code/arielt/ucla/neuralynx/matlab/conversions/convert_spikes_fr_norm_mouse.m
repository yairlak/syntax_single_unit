function [] = convert_spikes_fr_norm_mouse(time_slot_length, first_ch, ...
            last_ch, use_norm_fr)
% convert_spikes_fr_norm_mouse    

% Author: Ariel Tankus.
% Created: 01.07.2005.


if (nargin < 1)
    time_slot_length = 0.010;     % sec.
end

if (nargin < 3)
    first_ch = 1;
    last_ch  = get_num_channels;
end
if (nargin < 4)
    use_norm_fr = false;
end

convert_spike_times_to_fr(time_slot_length, first_ch, last_ch);
convert_fr_to_fr_norm_mouse(first_ch, last_ch, use_norm_fr);
convert_mouse_rec_to_vel_all;

fprintf('Done.\n');
