function clock_sync(is_eeg, mouse_basename, cheetah_bit, ...
                    use_existing_scaling, scaling_mouse_basename)
% clock_sync    Synchronize the clock of Cheetah with the clock of
%               another computer: paradigm computer, central computer,
%               EEG, etc.
%
%               clock_sync(is_eeg, mouse_basename, cheetah_bit)
%               is_eeg  - 1x1 - logical - true iff the other clock is the
%                                         the Stellate EEG recording.
%               mouse_basename - string - Base file name of the events
%                                         recording.
%               cheetah_bit - 1x1 - {0,...,15} - Bit number to which TTLs
%                                         are assigned.  To use a manual file of
%                                         Cheetah TTL times, use higher bits
%                                         (>=16).

% Author: Ariel Tankus.
% Created: 24.04.2009.

addpath(genpath('../../'))
addpath(genpath('../../../../../Spike_sorting/Neuralynx_pipeline/'))

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
    cheetah_bit = 0;
end

mouse_fname = [mouse_basename, '.log'];
new_mouse_fname = get_fname_w_timestamp_multiple(mouse_fname);

for i=1:length(new_mouse_fname)
    
    fprintf('Trying to sync file #%d: %s\n\n', i, new_mouse_fname{i});
    
    if (~strcmp(new_mouse_fname{i}, mouse_fname))
        [s, w] = unix(sprintf('ln -s %s %s', new_mouse_fname{i}, mouse_fname));
        if (s > 0)
            error('Error linking %s to %s:\n%s', new_mouse_fname{i}, mouse_fname, w);
        end
    end

    try
        [s, w] = unix(sprintf('./recording_to_relative_times %s', mouse_basename));
        if (s > 0)
            error('Error executing recording_to_relative_times %s:\n%s', ...
                  mouse_basename, w);
        end
        
        if ((nargin >=4) && (use_existing_scaling == true))
            % don't synchronize by TTLs, but use an already-computed shift+scaling:
            if (nargin < 5)
                scaling_mouse_basename = mouse_basename;
            end
            load([scaling_mouse_basename, '_time_scale_coeffs.mat']);    % will load
                                                                         % coefficients `b'.
            apply_linear_time_scaling(b, is_eeg, mouse_basename);
            
            return;
        end 
        
        load_nev_per_bit;
        
        cheetah_neuroball_time_offset_and_interp(is_eeg, mouse_basename, cheetah_bit);
        if (is_eeg)
            % for EEG, need also to match times of mouse recording to Cheetah:
            cheetah_neuroball_time_offset_and_interp(false);
        end
        
        fprintf('\nSync with file #%d: %s\n\n', i, new_mouse_fname{i});
        return;
    
    catch ME
        
        fprintf('%s\n\n\n', getReport(ME));
        
        if (~strcmp(new_mouse_fname{i}, mouse_fname))
            delete(mouse_fname);     % remove the link
        end
    end

    
end
