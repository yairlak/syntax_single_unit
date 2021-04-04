function [] = process_spike_times_multi_files(file_list)
% process_spike_times_multi_files    Synchronize a series of mouse recording
%                                    files with Cheatah's time.
%                                    The output is a unified
%                                    mouse_recording_in_cheetah_clock.log file.
%
%                                    process_spike_times_multi_files(file_list)
%                                    file_list - cell - nx1 or 1xn - list of
%                                                       file names, each is a
%                                                       mouse_recording.log
%                                                       file.
%
%                                    See also: process_spike_times.

% Author: Ariel Tankus.
% Created: 24.07.2007.


infile  = 'mouse_recording.log';
outfile = 'mouse_recording_in_cheetah_clock.log';
cat_output_cmd = 'cat';

% make sure none of the inputs is identical to infile:
for i=1:length(file_list)
    if (strcmp(file_list{i}, infile))
        movefile(infile, '_mouse_recording.log_TEMP');
        file_list{i} = '_mouse_recording.log_TEMP';
        break;
    end
end

for i=1:length(file_list)

    fprintf('File = %s\n', file_list{i});

    % use the symbolic name of infile for the current file:
    if (exist(infile, 'file'))
        delete(infile);
    end
    [s, w] = unix(sprintf('ln -s %s mouse_recording.log', file_list{i}));
    if (s > 0)
        error(sprintf('Unable to link %s to mouse_recording.log.', ...
                    file_list{i}));
    end

    % do actual work:
    [s,w] = unix('gen_neuroball_times');
    if (s > 0)
        error('Error executing gen_neuroball_times.');
    end

    cheetah_neuroball_time_offset_and_interp;

    % rename output:
    tmp_out_name = sprintf('%s_%d', outfile, i);
    movefile(outfile, tmp_out_name);
    cat_output_cmd = [cat_output_cmd, ' ', tmp_out_name];
end

% create a joint output file:
[s,w] = unix([cat_output_cmd, ' >! ', outfile]);
if (s > 0)
    error(sprintf('Error creating %s.', outfile));
end

% initial processing of the unified output:
[s,w] = unix('split_to_mouse_files_continuous');
if (s > 0)
    error('Error executing split_to_mouse_files_continuous.');
end
