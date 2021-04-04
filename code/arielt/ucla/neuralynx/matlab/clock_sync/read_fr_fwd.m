function [] = read_fr_fwd(filename)
% read_fr_fwd    

% Author: Ariel Tankus.
% Created: 30.12.2008.


fid = fopen(filename, 'r');
if (fid == -1)
    error(sprintf('Cannot open %s.', filename));
end

i = 1;
while (~feof(fid))
    [str, count] = fscanf(fid, '%s\n', 1);
    fprintf('%d %d %s\n', i, count, str);
    i = i + 1;
end

fclose(fid);
