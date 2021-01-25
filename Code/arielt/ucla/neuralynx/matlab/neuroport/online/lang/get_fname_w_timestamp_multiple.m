function new_fname = get_fname_w_timestamp_multiple(fname)
% get_fname_w_timestamp    Get a similar file name, that actually exists.
%                          This is useful when a timestamp exists in the
%                          file name, and only the base name is provided.
%                          This function will find the file (including the
%                          timestamp part).  This function requires that
%                          only one matching file exists.
%
%                          Allow multiple file names.
%
%                          new_fname = get_fname_w_timestamp_multiple(fname)
%                          fname - string - complete file name (e.g.:
%                                           sound.mp3).
%                          new_fname - string - a file name for a file which
%                                           actually exists.  It can be
%                                           either the original file (e.g.,
%                                           sound.mp3 if exists), or a
%                                           similar file (e.g.,
%                                           sound_2014-01-07_13-22-29.mp3).
%
%                          See also: read_sound_w_times.

% Author: Ariel Tankus.
% Created: 30.01.2014.


if (exist(fname))
    new_fname = {fname};
    return;
end


dot_inds = strfind(fname, '.');
if (isempty(dot_inds))
    % no suffix for file name.
    error('No suffix in file name: %s', fname);
end

fname_suffix = fname((dot_inds(end)+1):end);

% look for same file, but with a timestamp:
new_name = sprintf('%s_*-*-*_*-*-*.%s', fname(1:(end-4)), fname_suffix);
d = dir(new_name);
if (length(d) == 0)
    % cannot infer a similar file name:
    error('File not found: %s (no file: %s exist)', fname, new_name);
end

% file exists:
new_fname = {d(:).name};
