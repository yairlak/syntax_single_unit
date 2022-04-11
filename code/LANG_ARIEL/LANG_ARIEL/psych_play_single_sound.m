function psych_play_single_sound(freq_hz)
% BasicSoundOutputDemo([repetitions=0][, wavfilename])
%
% Demonstrates very basic use of the Psychtoolbox sound output driver
% PsychPortAudio(). PsychPortAudio is a better, more reliable, more accurate
% replacement for the old Psychtoolbox SND() function and other means of
% sound output in Matlab like sound(), soundsc(), wavplay(), audioplayer()
% etc.
%
% This demo only demonstrates normal operation, not the low-latency mode,
% extra demos and tests for low-latency and high precision timing output will
% follow soon. If you need low-latency, make sure to read "help
% InitializePsychSound" carefully or contact the forum.
% Testing for low-latency mode showed that sub-millisecond accurate sound
% onset and < 10 msecs latency are possible on Linux, OSX and on some specially
% configured MS-Windows ASIO sound card setups.
%
%
% Optional arguments:
%
% repetitions = Number of repetitions of the sound. Zero = Repeat forever
% (until stopped by keypress), 1 = Play once, 2 = Play twice, ....
%
% wavfilename = Name of a .wav sound file to load and playback. Otherwise
% the good ol' handel.mat file (part of Matlab) is used.
%
% The demo just loads and plays the soundfile, waits for a keypress to stop
% it, then quits.

% History:
% 06/07/2007 Written (MK)

% Running on PTB-3? Abort otherwise.
%AssertOpenGL;

if (nargin < 1)
    freq_hz = 400;
end

sampling_freq_hz = 44100;
rec_len_sec = 0.200;
wavedata = create_beep_waveform(freq_hz, rec_len_sec, sampling_freq_hz);

% Make sure we have always 2 channels stereo output.
% Why? Because some low-end and embedded soundcards
% only support 2 channels, not 1 channel, and we want
% to be robust in our demos.
wavedata = formatAudioForPsychToolbox(wavedata);
nrchannels = 2;

% Perform basic initialization of the sound driver:
InitializePsychSound;

% Open the default audio device [], with default mode [] (==Only playback),
% and a required latencyclass of zero 0 == no low-latency mode, as well as
% a frequency of freq and nrchannels sound channels.
% This returns a handle to the audio device:
try
    % Try with the 'freq'uency we wanted:
    pahandle = PsychPortAudio('Open', [], [], 0, sampling_freq_hz, nrchannels);
catch
    % Failed. Retry with default frequency as suggested by device:
    fprintf('\nCould not open device at wanted playback frequency of %i Hz. Will retry with device default frequency.\n', sampling_freq_hz);
    fprintf('Sound may sound a bit out of tune, ...\n\n');

    psychlasterror('reset');
    pahandle = PsychPortAudio('Open', [], [], 0, [], nrchannels);
end

% Fill the audio playback buffer with the audio data 'wavedata':
PsychPortAudio('FillBuffer', pahandle, wavedata);

% Start audio playback for 'repetitions' repetitions of the sound data,
% start it immediately (0) and wait for the playback to start, return onset
% timestamp.
repetitions = 1;
t1 = PsychPortAudio('Start', pahandle, repetitions, 0, 1);
log_file_writer(t1, sprintf('SOUND_ONSET %d %.3f %d', freq_hz, rec_len_sec, ...
                            sampling_freq_hz));

% Stay in a little loop until keypress:
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

% Stop playback:
PsychPortAudio('Stop', pahandle);

% Close the audio device:
PsychPortAudio('Close', pahandle);
