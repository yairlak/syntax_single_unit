function smoothed_table = smooth_start_stop_table(start_stop_table, ...
            time_slot_length, gauss_1sig_ms)
% smooth_start_stop_table    

% Author: Ariel Tankus.
% Created: 09.02.2006.


mu = 0;
sigma = gauss_1sig_ms ./ (time_slot_length.*1000);   % std. dev.; units: #bins.
wave_x = (-3*sigma):(3*sigma);

smoothed_table = zeros(size(start_stop_table));

for i=1:size(start_stop_table, 1)
    smoothed_table(i, :) = conv2(double(start_stop_table(i, :)), ...
                gauss(wave_x, sigma, mu), 'same');
end
