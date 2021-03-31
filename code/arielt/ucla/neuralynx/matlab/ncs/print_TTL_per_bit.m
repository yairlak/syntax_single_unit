function [] = print_TTL_per_bit(TTL_per_bit)
% print_TTL_per_bit    

% Author: Ariel Tankus.
% Created: 18.01.2009.


for i=1:length(TTL_per_bit)
    filename = sprintf('cheetah_event_times_bit%d.log', i-1);
    fid = fopen(filename, 'w');
    for j=1:length(TTL_per_bit{i})
        fprintf(fid, '%s\n', num2str(TTL_per_bit{i}(j)));
    end
    fclose(fid);
end
