function c = read_complete_mouse_recording(filename)
% read_complete_mouse_recording    Read a complete mouse recording file (of
%                                  NeuroBall) into a time stamp and a string
%                                  containing the rest of the line.
%
%                                  filename - string - file name.
%                                  c - cell {1, 2} - c{1} - vector of time
%                                                    stamps.  c{2} - vector
%                                                    of strings (the rest of
%                                                    the event line).
%
%                                  See also: textscan.

% Author: Ariel Tankus.
% Created: 15.03.2005.

fid = fopen(filename, 'r');
c = textscan(fid, '%n %[^\n]\n');
fclose(fid);

if (isempty(c{1}))
    error('Empty paradigm recording: %s!', filename);
end
