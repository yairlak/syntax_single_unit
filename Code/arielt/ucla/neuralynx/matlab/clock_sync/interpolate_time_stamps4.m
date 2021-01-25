load ~/ucla/analysis4/cheetah_times_relative.txt
load ~/ucla/analysis4/neuroball_times_relative.txt


b = regress(cheetah_times_relative(75:339), ...
            [neuroball_times_relative(3:267), ones(339-75+1, 1)]);

c = read_complete_mouse_recording('mouse_recording_relative.log');

% express NeuroBall event times in Cheetah's clock:
c{1} = b(1).*c{1} + b(2);
c{1} = floor(c{1});    % c{1} is in microseconds, so the fraction is
                       % unnecessary.

fid = fopen('mouse_recording_in_cheetah_clock.log', 'w');
for i=1:length(c{2})
    fprintf(fid, '%d %s\n', c{1}(i), c{2}{i});
end
fclose(fid);
