function pahandle = init_psych_sound_system(audio_fname_cell, files_subdir)
% init_psych_sound_system    

% Author: Ariel Tankus.
% Created: 16.03.2017.


reallyneedlowlatency = true;

global y_cell;
global freq_array;
global nrchannels;

% Running on PTB-3? Abort otherwise.
AssertOpenGL;

num_files = length(audio_fname_cell);

y_cell     = cell(num_files, 1);
freq_array = NaN(num_files, 1);
nrchannels = NaN(num_files, 1);

% read and store all sound files:
for i=1:num_files
    % Read WAV file from filesystem:
    [files_subdir, filesep, audio_fname_cell{i}]
    [y, freq] = audioread([files_subdir, filesep, audio_fname_cell{i}]);
    y_cell{i} = y';
    freq_array(i) = freq;
    
    nrchannels(i) = size(y_cell{i}, 1); % Number of rows == number of channels.
end

InitializePsychSound(reallyneedlowlatency);

% Open the default audio device [], with default mode [] (==Only playback),
% and a required latencyclass of zero 0 == no low-latency mode, as well as
% a frequency of freq and nrchannels sound channels.
% This returns a handle to the audio device:
try
    % Try with the 'freq'uency we wanted:
    pahandle = PsychPortAudio('Open', [], [], 0, freq_array(1), nrchannels(1));
catch
    % Failed. Retry with default frequency as suggested by device:
    fprintf('\nCould not open device at wanted playback frequency of %i Hz. Will retry with device default frequency.\n', freq);
    fprintf('Sound may sound a bit out of tune, ...\n\n');

    psychlasterror('reset');
    pahandle = PsychPortAudio('Open', [], [], 0, [], nrchannels);
end
