function [t1, t_end] = psych_play_sound_file(pahandle, audio_fname_cell, ...
                                             audio_file_ind, playback_start_time)
% psych_play_sound_file    

% Author: Ariel Tankus.
% Created: 16.03.2017.


global y_cell;
global freq_array;
global nrchannels;

% Fill the audio playback buffer with the audio data 'wavedata':
PsychPortAudio('FillBuffer', pahandle, y_cell{audio_file_ind});

repetitions  = 1;
waitForStart = 1;
t1 = PsychPortAudio('Start', pahandle, repetitions, playback_start_time, ...
                    waitForStart);
log_file_writer(t1, sprintf('AUDIO_PLAYBACK_ONSET %s', ...
                            audio_fname_cell{audio_file_ind}));

% Stay in a little loop until playback is over:
while true
    % Wait a seconds...
    WaitSecs(0.010);
    
    % Query current playback status and print it to the Matlab window:
    s = PsychPortAudio('GetStatus', pahandle);
    % tHost = GetSecs;

    if (~s.Active)
        break;
    end
end

s = PsychPortAudio('GetStatus', pahandle);
t_end = s.EstimatedStopTime;
log_file_writer(t_end, sprintf('AUDIO_PLAYBACK_DONE %s', ...
                               audio_fname_cell{audio_file_ind}));
