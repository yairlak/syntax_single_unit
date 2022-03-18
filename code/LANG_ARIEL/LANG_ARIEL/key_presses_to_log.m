function key_presses_to_log()
% key_presses_to_log    

% Author: Ariel Tankus.
% Created: 31.01.2017.


global is_exiting;

defKeyboard;

[kb_event, kb_nremaining] = KbEventGet();
while (~isempty(kb_event))
    log_file_writer(kb_event.Time, ...
                    sprintf('KEY_PRESS %s', KbName(kb_event.Keycode)));

    if (kb_event.Keycode == exitKey)
        log_file_writer(GetSecs, sprintf('QUIT'));  % actual time of quit.
        is_exiting = true;
        return;
    end
    
    if (kb_nremaining == 0)
        return;
    end
    [kb_event, kb_nremaining] = KbEventGet();
end
