function [] = load_nev_per_bit()
% load_nev_per_bit    

% Author: Ariel Tankus.
% Created: 18.01.2009.


TTL_per_bit = load_nev_multi;
print_TTL_per_bit(TTL_per_bit);
[s, w] = unix('if (!(-e manual_cheetah_events.txt)), ln -sf cheetah_event_times_bit0.log manual_cheetah_events.txt; endif');
if (s ~= 0)
    fprintf('Failed to create link manual_cheetah_events.txt .\n');
end
