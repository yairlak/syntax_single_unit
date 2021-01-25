%function [] = cmp_sync_cheetah()
% cmp_sync_cheetah    

% Author: Ariel Tankus.
% Created: 19.11.2015.


process_spike_times(false, 'sync_pulse_loop', 0, false, 'sync_pulse_loop');

load ./sync_pulse_loop_times_relative.txt
load ./cheetah_event_times_bit0.log

d_cheetah = diff(cheetah_event_times_bit0);
d_sync = diff(sync_pulse_loop_times_relative);

dd = d_cheetah - d_sync(3:end);

plot(dd(2:end));
title('dd = d_sync - d_cheetah;', 'Interpreter', 'none')
xlabel('sync pulse [#]')
ylabel('dd [microsec]', 'Interpreter', 'none')
print -dpng dd.png

rec_time_sync_micsec = sync_pulse_loop_times_relative(end)-sync_pulse_loop_times_relative(1)
rec_time_cheetah_micsec = cheetah_event_times_bit0(end)-cheetah_event_times_bit0(1)

save clock_sync_analysis.mat;
