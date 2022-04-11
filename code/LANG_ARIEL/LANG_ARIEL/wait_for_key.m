function wait_for_key(key_str)
% wait_for_key    Wait for a key press. Log the key when received.

% Author: Ariel Tankus.
% Created: 31.01.2017.



[~, ~, keyCode] = KbCheck;
log_key_press(keyCode);

% Wait for reaction
while ~sum(keyCode(key_str))                  % wait for answer
    [~, ~, keyCode] = KbCheck;
    log_key_press(keyCode);
end
