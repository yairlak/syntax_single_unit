function log_key_press(keyCode)
% log_key_press    keyCode - boolean vector for each key code. true iff
%                            corresponding key pressed.

% Author: Ariel Tankus.
% Created: 31.01.2017.


if (sum(keyCode))
    key_list = KbName(find(keyCode));
    if (ischar(key_list))
        log_file_writer(GetSecs, sprintf('KEY_PRESS %s', key_list));
    else
        % cell array of strings:
        for k=1:length(key_list)
            log_file_writer(GetSecs, sprintf('KEY_PRESS %s', key_list(k)));
        end
    end
end
