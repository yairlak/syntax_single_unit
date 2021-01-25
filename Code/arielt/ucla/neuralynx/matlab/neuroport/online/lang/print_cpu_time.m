function cpu_time_sec_total = print_cpu_time(start_time, num_iters)
% print_cpu_time    Print the total CPU time used since `start_time', and
%                   return this time in seconds.
%
%                   cpu_time_sec_total = print_cpu_time(start_time, num_iters)
%                   start_time - 1x1 - CPU start time, as was recorded at
%                                      the beginning of time interval by
%                                      `cputime'.
%                   cpu_time_sec_total - 1x1 - Total time in CPU seconds
%                                      since `start_time'.
%                   num_iters          - 1x1 - [opt] Number of iterations
%                                      performed since `start_time'.
%
%                   See also: cputime.

% Author: Ariel Tankus.
% Created: 16.06.2002.

cpu_time_sec_total = cputime - start_time;
cpu_time_hours = fix(cpu_time_sec_total/3600);
cpu_time_mins  = fix(rem(cpu_time_sec_total, 3600) / 60);
cpu_time_secs  = rem(cpu_time_sec_total, 60); 

fprintf('CPU time = %02d:%02d:', cpu_time_hours, cpu_time_mins);
if (cpu_time_secs < 10)
    % fprintf doesn't print leading 0s with %g:
%    fprintf('0');
    fprintf('0%1.5g  (%G secs.', cpu_time_secs, cpu_time_sec_total);
else
    fprintf('%2.5g  (%G secs.', cpu_time_secs, cpu_time_sec_total);
end
if (nargin >= 2)
    fprintf(', %d iters.', num_iters);
end
fprintf(');   ');

print_date;
fprintf('\n');
