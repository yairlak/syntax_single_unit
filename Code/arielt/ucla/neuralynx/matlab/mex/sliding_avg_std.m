function [avg_per_win, std_per_win] = sliding_avg_std(vec, win_length)
% sliding_avg_std    Computer the average and standard deviation of a vector
%                    in a sliding window.
%
%                    [avg_per_win, std_per_win] = sliding_avg_std(vec, win_length)
%                    vec - nx1 or 1xn - vector.
%                    win_length - 1x1 - #elements to include in the
%                                       computation (i.e., in a window).
%                    avg_per_win, std_per_win - (n-win_length+1)x1 - averages and
%                                       standard deviations in a window.

% Author: Ariel Tankus.
% Created: 28.05.2005.
% Modified: 20.11.2005.


if (win_length <= 0)
    error('win_length must be a natural number.');
end

num_wins = length(vec) - win_length + 1;
win_len_m1 = win_length - 1;
avg_per_win = zeros(num_wins, 1);
std_per_win = zeros(num_wins, 1);

sum_fr_win = sum(vec(1:(win_length-1)));

for i=win_length:length(vec)
    start_ind = i - win_length + 1;

    sum_fr_win = sum_fr_win + vec(i);
    avg_per_win(start_ind) = sum_fr_win ./ win_length;

    std_per_win(start_ind) = ...
        sqrt(sum((vec(start_ind:i) - avg_per_win(start_ind)).^2) ./ win_len_m1);

    sum_fr_win = sum_fr_win - vec(start_ind);
end
