function c = read_complete_fr_online(filename)
% read_complete_fr_online    Read a complete mouse recording file (of the
%                            online computation) into a local time stamp,
%                            a Cheetah time stamp and a string
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
c = textscan(fid, '%n %[^ ] %[^\n]\n');
fclose(fid);
