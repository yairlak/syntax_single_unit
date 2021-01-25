function [] = convert_mouse_rec_to_vel_all(conversion_type)
% convert_mouse_rec_to_vel_all    

% Author: Ariel Tankus.
% Created: 14.06.2005.


if (nargin < 1)
    conversion_type = 'FR';
end

in_basename    = 'mouse_level';
out_basename   = 'mouse_vel_accel';
list_file_name = 'mouse_vel_accel_file_list.txt';
with_filter    = false;
num_param_ranges = 11;   % when modifying num_param_ranges, also modify the
                          % default in get_kinematics_params_list.
ch_list = 1:get_num_channels;
func_name = 'convert_mouse_rec_to_vel';

convert_some_mouse_rec_to_vel(in_basename, out_basename, list_file_name, ...
            false, with_filter, num_param_ranges, ch_list, func_name, ...
            conversion_type);
