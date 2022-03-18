function [] = psych_play_seq_sounds()
% psych_play_seq_sounds    

% Author: Ariel Tankus.
% Created: 01.02.2017.


freq_hz = 400;
psych_play_single_sound(freq_hz);

WaitSecs(0.100);

freq_hz = 300;
psych_play_single_sound(freq_hz);

WaitSecs(0.400);

freq_hz = 600;
psych_play_single_sound(freq_hz);

WaitSecs(0.200);

freq_hz = 500;
psych_play_single_sound(freq_hz);

freq_hz = 700;
psych_play_single_sound(freq_hz);

WaitSecs(0.150);

freq_hz = 300;
psych_play_single_sound(freq_hz);

WaitSecs(0.200);

freq_hz = 400;
psych_play_single_sound(freq_hz);

WaitSecs(0.150);

freq_hz = 300;
psych_play_single_sound(freq_hz);
