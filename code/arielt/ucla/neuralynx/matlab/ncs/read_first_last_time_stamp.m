function [first_time_stamps, last_time_stamp] = ...
    read_first_last_time_stamp(filename, force_recalc)
% read_first_last_time_stamp    Read the first, second and last time stamps
%                               in a CSC Cheatah recording file.
%
%                               [first_time_stamps, last_time_stamp] = ...
%                                    get_first_time_stamp(filename)
%                               filename - string - .Ncs file name.
%                               force_recalc - 1x1 - logical - true: force
%                                          recomputation even if cache exists.
%                                          false: use cache if exists (default).
%                               first_time_stamps - 1x2 - first and second
%                                          time stamps in the file.
%                               last_time_stamp   - 1x1 - lasst time stamps
%                                          in the file.
%     
%                               See also: read_ncs, convert_ncs,
%                                         read_first_time_stamp.

% Author: Ariel Tankus.
% Created: 05.12.2005.


if (nargin < 2)
    force_recalc = false;
end

if ((force_recalc) || (~exist(['.', filesep, 'time_stamps_cache.mat'], 'file')))

    if (exist('./electrode_info.mat', 'file'))
        % Neuroport:
        load electrode_info.mat;
        first_time_stamps = [0; 1E6./samp_freq_hz];
        last_time_stamp   = (num_samples-1).*1E6./samp_freq_hz;
    else
        % Neuralynx:
        f = fopen(filename);
        if (f == -1)
            error(['Cannot open ', filename]);
        end
        fseek(f, 16384, 'bof');         % Skip Header, put pointer to the first record.
                                        % Read first and second time stamps:
        first_time_stamps = fread(f, 2, 'int64', (4+4+4+2*512));
    
        fseek(f, -(8+4+4+4+2*512), 'eof');         % Return 1 record back from eof.
        last_time_stamp = fread(f, 1, 'int64');
        fclose(f);
    end
    
    save time_stamps_cache first_time_stamps last_time_stamp;
    
else
    
    % read from cache
    load time_stamps_cache;

end
