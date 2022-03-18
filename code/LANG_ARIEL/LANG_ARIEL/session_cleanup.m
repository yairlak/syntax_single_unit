function [] = session_cleanup()
% session_cleanup    

% Author: Ariel Tankus.
% Created: 30.01.2017.
global ttlLog

log_file_writer('close'); 

KbQueueStop();

sca;
ShowCursor;
fclose('all');
Priority(0);
diary off

str = sprintf('ttlLog_syntax_%s',datestr(now,'yyyy-mm-dd_HH-MM'));
disp(str)
save(str,'ttlLog')
