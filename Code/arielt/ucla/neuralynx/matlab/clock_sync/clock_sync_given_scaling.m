function clock_sync_given_scaling(mouse_base, scaling_mouse_basename)
% clock_sync_given_scaling    

% Author: Ariel Tankus.
% Created: 16.01.2013.


if (nargin < 1)
    mouse_base = 'mouse_recording';
end
if (nargin < 2)
    scaling_mouse_basename = 'sync_pulse_loop';
end

is_eeg = false;
mouse_basename = mouse_base;
cheetah_bit = 0;
use_existing_scaling = true;

clock_sync(is_eeg, mouse_basename, cheetah_bit, ...
           use_existing_scaling, scaling_mouse_basename);
