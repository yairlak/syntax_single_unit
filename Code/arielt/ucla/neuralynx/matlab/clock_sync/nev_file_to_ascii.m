function nev_file_to_ascii(outfile, nev_file)
% nev_file_to_ascii    Convert a .Nev file into an ASCII file with the
%                      times of the TTLs ascendingly sorted.
%                      This may be used to manually load the list into gnumeric.
%
%                      nev_file_to_ascii(outfile, nev_file)
%                      outfile  - string - Output text file name.
%                      nev_file - string - Input .Nev file name. [default:
%                                          Events.Nev]

% Author: Ariel Tankus.
% Created: 13.12.2008.


if (nargin < 2)
    nev_file = 'Events.Nev';
end

[upstrokes,downstrokes]=load_nev(nev_file);
a = sort([upstrokes'; downstrokes']);
a_with_newline = [int2str(a), repmat(sprintf('\n'), size(a, 1), 1)];

fid = fopen(outfile, 'w');
fprintf(fid, '%s', a_with_newline');
fclose(fid);
