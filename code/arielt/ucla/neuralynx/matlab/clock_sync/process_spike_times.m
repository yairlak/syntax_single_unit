function [] = process_spike_times(is_eeg, mouse_basename, cheetah_bit, use_existing_scaling, scaling_mouse_basename)
% process_mouse_and_spike_times    

% Author: Ariel Tankus.
% Created: 14.08.2005.
% Modified: 24.04.2009.  mouse_basename and cheetah_bit added for central
%                        computer processing.
% Modified: 07.09.2017. Avoid running `split_to_mouse_files_continuous' if
%                       not a mouse_recording.log.


cheetah_bit_fname = './cheetah_bit.mat';

if (nargin < 1)
    is_eeg = false;
end
if (nargin < 2)
    if (~is_eeg)
        mouse_basename = 'mouse_recording';
    else
        mouse_basename = 'eeg';
    end
end
if (nargin < 3)
    if (exist(cheetah_bit_fname, 'file'));
        load(cheetah_bit_fname);
    else
        cheetah_bit = 0;
    end
    fprintf('Using Cheetah bit %d.\n', cheetah_bit);
end
if (nargin  < 4)
    use_existing_scaling = false;
end
if (nargin < 5)
    scaling_mouse_basename = 'mouse_recording';
end

clock_sync(is_eeg, mouse_basename, cheetah_bit, use_existing_scaling, scaling_mouse_basename);

if (strcmp(mouse_basename, 'mouse_recording'))
    [s,w] = unix('split_to_mouse_files_continuous');
    if (s > 0)
        error('Error executing split_to_mouse_files_continuous.');
    end
end
