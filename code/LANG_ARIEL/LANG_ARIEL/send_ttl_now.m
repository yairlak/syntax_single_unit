function send_ttl_now(event_code)
% send_ttl_now    

% Author: Ariel Tankus.
% Created: 30.01.2017.


global sendTTL;
global TTL; 
global ttl;
global location;
global dio;
global serial_port;
global ttlLog;
global portA;

if ~sendTTL
    return;
end


if strcmp(location,'UCLA')
    message =num2str(event_code);
    eyelink = [];
    
    [ttlLog, pre_send_time, post_send_time] = sendTTL_em(message,serial_port, dio, eyelink,ttlLog);
    timeSinceStart = pre_send_time;
%    DaqDOut(dio, portA, event_code);
%    WaitSecs(ttlwait);
%    DaqDOut(dio, portA, eventreset);
elseif strcmp(location,'TLVMC')
    pre_send_time = GetSecs;
    fwrite(sio, event_code);
    post_send_time = GetSecs;
else
    return;
end

ttl(TTL,1) = post_send_time(1);
ttl(TTL,2) = event_code;
TTL = TTL + 1;

for i=1:length(post_send_time)
    log_file_writer(post_send_time(i), ...
                    sprintf('CHEETAH_SIGNAL SENT_AFTER_TIME %.6f', ...
                            pre_send_time(i)));
end
