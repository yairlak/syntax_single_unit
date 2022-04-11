function log_file_writer(event_time, event_str)
% log_file_writer    
%
%                    log_file_writer(event_time, event_str);
%                    log_file_writer('open', fname);
%                    log_file_writer('close');

% Author: Ariel Tankus.
% Created: 30.01.2017.


%num_log_events = 1000;
num_log_events = 10;    % Events correspond to stimuli, s.t. we don't loose
                        % much in case of a software failure.
persistent is_open;
persistent fid;
persistent event_times;
persistent event_strs;
persistent cur_event_ind;


if (isstr(event_time))
    if (strcmp(event_time, 'open'))
    
        fname = [event_str, '_', datestr(now, 'yyyy-mm-dd_HH-MM-SS'), '.log'];
        fid = fopen(fname, 'w');
        if (fid == -1)
            error('Opening log file %s failed.', fname);
        end
        if (isunix)
            [pathstr,base_fname,ext] = fileparts(fname);
            [s, w] = unix(sprintf('ln -sf %s.log %s.log', base_fname, event_str));
            if (s ~= 0)
                fprintf('WARNING: Failed to create symbolic link: %s\n', w);
            end
        end
        is_open = true;
        event_times = NaN(num_log_events, 1);
        event_strs  = cell(num_log_events, 1);
        cur_event_ind = 1;
    
    elseif (strcmp(event_time, 'close'))

        if (is_open)
            fprintf('Closing log_file writer.\n');

            % write all pending events:
            for i=1:(cur_event_ind-1)
                fprintf(fid, '%.6f %s\n', event_times(i), event_strs{i});
            end
            
            fclose(fid);
            is_open = false;
        end
        
    else
        error('Unknown event_time string ''%s'' encountered.', event_time);
    end

    return;
end

if (~is_open)
    fprintf('Write to a closed log file FAILED.\n');
    return;
end

event_times(cur_event_ind) = event_time;
event_strs{cur_event_ind}  = event_str;
cur_event_ind = cur_event_ind + 1;

if (cur_event_ind > num_log_events)
    for i=1:num_log_events
        fprintf(fid, '%.6f %s\n', event_times(i), event_strs{i});
    end

    % buffer is logically cleared:
    cur_event_ind = 1;
end
