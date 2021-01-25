function print_date
% print_date    Pretty print current date and time.
%
%               See also: now, datevec.

% Author: Ariel Tankus.
% Created: 07.07.2001.

time_of_end = floor(datevec(now));
fprintf('%02d-%02d-%04d %02d:%02d:%02d', time_of_end([3:-1:1, 4:6]));
fprintf('\n');
