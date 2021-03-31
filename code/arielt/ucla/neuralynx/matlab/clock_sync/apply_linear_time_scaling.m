function [] = apply_linear_time_scaling(b, is_eeg, mouse_basename)
% apply_linear_time_scaling    Apply a given linear transformation (shift +
%                              scaling) to a given mouse recording file.

% Author: Ariel Tankus.
% Created: 14.01.2013.


if (~is_eeg)
    linear_scale_mouse_recording(b, mouse_basename);
else
    load channels;
    eeg_to_csc_files(ch_list, pwd, 'eeg.mat', b);
end
