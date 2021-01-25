function [] = convert_some_mouse_rec_to_vel(in_basename, out_basename, ...
            list_file_name, common_intervals_file, with_filter, ...
            num_param_ranges, ch_list, func_name, conversion_type)
% convert_mouse_rec_to_vel_all    

% Author: Ariel Tankus.
% Created: 14.06.2005.


if (nargin < 5)
    with_filter = false;
end
if (nargin < 6)
    num_param_ranges = 10;
end
if (nargin < 7)
    ch_list = 1:get_num_channels;
end
if (nargin < 8)
    func_name = 'convert_mouse_rec_to_vel';
end

% file which lists all mouse_vel_accel{}_exclude_display files.
fid = fopen(list_file_name, 'w');
if (fid == -1)
    error(sprintf('Cannot open file: %s for write.', list_file_name));
end

switch (lower(conversion_type))
 case 'fr'
  load CSC1_fr_norm;     % this is necessary only for: first_edge, last_edge,
                         % time_slot_length.
 case {'mua', 'lfp'}
  load(sprintf('CSC1_%s.mat', lower(conversion_type)));
  first_edge = time_first_sec - low_sr_time_interval./2;
  last_edge  = time_last_sec  + low_sr_time_interval./2;
  time_slot_length = low_sr_time_interval;
  
 case 'eeg'
  load(sprintf('CSC1_eeg.mat'));
  
end

mouse_recording_file = [in_basename, '_all_mouse_only.log'];
non_display_intervals_file = [in_basename, '_all_intervals.log'];
if (exist(mouse_recording_file, 'file'))
    feval(func_name, mouse_recording_file, non_display_intervals_file, ...
         out_basename, ch_list, with_filter, num_param_ranges, first_edge, ...
          last_edge, time_slot_length);
%    fprintf(fid, '%s_exclude_display\n', out_basename);
end

level = 1;
mouse_recording_file = sprintf('%s%d_mouse_only.log', in_basename, level);
while (exist(mouse_recording_file, 'file'))

    if (~common_intervals_file)
        non_display_intervals_file = sprintf('%s%d_intervals.log', in_basename, ...
                    level);
    end
    cur_out_basename = sprintf('%s%d', out_basename, level);

    ok = feval(func_name, mouse_recording_file, non_display_intervals_file, ...
               cur_out_basename, ch_list, with_filter, num_param_ranges, ...
               first_edge, last_edge, time_slot_length);
    if (ok)
        fprintf(fid, [cur_out_basename, '_exclude_display\n']);
    end
    
    level = level + 1;
    mouse_recording_file = sprintf('%s%d_mouse_only.log', in_basename, level);
end

fclose(fid);
