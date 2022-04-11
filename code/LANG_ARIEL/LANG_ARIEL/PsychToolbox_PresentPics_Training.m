%% This function has 3 parts - 
%% explaing task, show stimuli, test presented stimuli only

%% Run the series of explain task --> train --> test through : 
%% /Users/mayags/Documents/LA_paradigms/MemLocPairing/MemLocWrapper_RUN_ME.m

%% See similar example in - 
%% /Applications/Psychtoolbox/PsychDemos/PsychExampleExperiments/OldNewRecognition/OldNewRecogExp.m

% PsychToolbox_PresentPics_Training

global event_list
global sendTTL;
%global TTL; 
%global ttl;
global location;
global event_track; 
global debug;
global dio;
global portA;
global ttlLog
%defKeyboard();
DefTTLs()

log_dir = sprintf('log_patient');

% % dio = 3;
% % % dio=DaqDeviceIndex;                                     	% get a handle for the USB-1208FS
% % hwline=DaqDConfigPort(dio,0,0);                            	% configure digital port A for output
% % DaqDOut(dio,0,eventreset);
% % hwline=DaqDConfigPort(dio,1,0);                          	% configure digital port B for output
% % DaqDOut(dio,1,eventreset); laststim = 0;

% %% Prepare training and testing folders 
% comp_n = getenv('COMPUTERNAME');
% if isunix
%     !hostname > hostname.txt;
%     hostName = textread('hostname.txt','%s');
% end
% if exist('hostName','var') && strfind(hostName{1},'Nir'); % MAC ENV
%     !echo $USER > username.txt
%     userName = textread('username.txt','%s');
%     if (strfind(userName{1},'maya'))
%         file_folder_main = '/Users/mayags/Dropbox/Code/Nir Lab/Stimuli';
%     else
%         file_folder_main = '/Users/nirlab/Documents/MATLAB/Maya/Stimuli';
%     end
%     
% elseif (strcmp(comp_n,'SLP6')) % Windows ENV
%     file_folder_main = 'C:\Maya\Dropbox\Code\Nir Lab\Stimuli\';
% end

file_folder_main = pwd;
file_folder_Q = fullfile(file_folder_main, stimuli_subdir);
log_dir = fullfile(file_folder_main, log_dir);
if isempty(dir(log_dir))
    mkdir(log_dir);
end

load(filename_stimuli,'params')
if (~params.stimuli_in_text_file)
    num_stimuli = length(params.stimuliList);
end

% Update main struct
Stimuli_presentation.start_time = datestr(now);
Stimuli_presentation.logfile_dir = log_dir;
Stimuli_presentation.event_list = event_list;

%filename = fullfile(log_dir,sprintf('ONLINE_training_results_%f.mat',now));
%save(filename,'Stimuli_presentation','ttl','TTL')
     

% Now matlab cannot accept key-strokes
% To exit this status - hit ctrl-c several times --> cmd-0 --> sca -->
% return
if ~debug
    HideCursor;
    ListenChar(2);
end

try
%% Initialize screen
screens = Screen('Screens');
whichScreen = max(screens);                                                 % Selects the highest screen number as your output screen
% whichScreen = 0;                                                          % Selects the laptop screen

Screen('Preference','VisualDebugLevel',1);% Switch off your screen check
isSkipTest = 1;
Screen('Preference','SkipSyncTests',isSkipTest);

white = WhiteIndex(whichScreen);                                            % Returns the mean value of white at the current screen depth
black = BlackIndex(whichScreen);                              	            % Returns the mean value of black at the current screen depth
gray = GrayIndex(whichScreen);                                          	% Returns the mean value of gray at the current screen depth

[window,windowRect] = Screen('OpenWindow',whichScreen, white);

blackscreen = Screen('MakeTexture', window, black);
Screen('DrawTexture', window, blackscreen);
Screen('Flip', window);
Screen(window,'TextSize',50);

winRect = Screen('Rect', window); % [0,0,1440,900] for the Nir-lab mac screen

FlushEvents('keyDown');

% Set priority for script execution to realtime priority:
priorityLevel=MaxPriority(window);
Priority(priorityLevel);

   
% initialize KbCheck and variables to make sure they're
% properly initialized/allocted by Matlab - this to avoid time
% delays in the critical reaction time measurement part of the
% script:
[KeyIsDown, endrt, KeyCode]=KbCheck;

KbQueueCreate();
KbQueueStart();

%%% --------------------------------------------------------------
% PartA - First show an explanation slide - the response is a space key
instructions_fname = fullfile(params.instructions);
Image = imread(instructions_fname);   % Load instructions image

im_fixation = imread(params.fixation);   % Load fixation image

send_ttl_now(START_SEC);

PresentExplanationSlide;

send_ttl_now(END_SEC);


%%% --------------------------------------------------------------
% PartB - learn stimuli in different screen locations
send_ttl_now(START_SEC);

if (~params.is_audio)
    if (~params.stimuli_in_text_file)
        % read all images:
        im_array    = cell(num_stimuli, 1);
        fname_array = cell(num_stimuli, 1);
        for ii_S = 1:num_stimuli
            fname_array{ii_S} = fullfile(params.stimuli_subdir, params.stimuliList{ii_S});
            im_array{ii_S} = imread(fname_array{ii_S});   % Load an image
        end
    else
        % read text file:
        % read each line as 1 string:
        text_line_array = textread(['Stimuli/',params.stimuliList], '%s', 'delimiter', '\n');
        num_lines = length(text_line_array);
        line_serial_no = (1:num_lines)';
        if (params.use_rand_perm)
            p = randperm(num_lines);
            text_line_array = text_line_array(p);
            line_serial_no  = line_serial_no(p);
        end
        sentence_split_to_words = cell(num_lines, 1);
        num_words_per_sentence = NaN(num_lines, 1);
        word_array = {};
        word_num_in_sentence = [];
        for i=1:num_lines
            sentence_split_to_words{i} = strread(text_line_array{i}, '%s');
            num_words_per_sentence(i) = length(sentence_split_to_words{i});
            word_array = [word_array; sentence_split_to_words{i}];
            word_num_in_sentence = [word_num_in_sentence;
                                    [repmat(line_serial_no(i), num_words_per_sentence(i), 1), (1:num_words_per_sentence(i))']];
        end
        sentences_start = cumsum([1; num_words_per_sentence(1:(end-1))]);
        num_stimuli = length(word_array);
    end
end

% FIXATION:
tex_fixation = Screen('MakeTexture', window, im_fixation);
Screen('DrawTexture', window, tex_fixation);
vbloff = Screen('Flip', window); % Update view --> image is off

send_ttl_now(IMAGE_OFF);
log_file_writer(vbloff, 'DISPLAY_FIXATION');

rect = get(0, 'ScreenSize');
rect = [0 0 rect(3:4)];

if (params.is_audio)
    % initialize audio system:
    files_subdir     = params.stimuli_subdir;
    audio_fname_cell = params.stimuliList;
    pahandle = init_psych_sound_system(audio_fname_cell, files_subdir);
    
    t_end = -POST_AUDIO_PLAYBACK_SEC;  % schedule next playback to time 0
                                       % (t_end + POST_AUDIO_PLAYBACK_SEC).
end

if (params.stimuli_in_text_file)
    % Select specific text font, style and size:
%    Screen('TextFont',window, 'Courier New');
    Screen('TextFont',window, 'Arial');
    Screen('TextSize',window, 160);   % 160 --> ~25mm text height (from top
                                      % of `d' to bottom of `g').
    Screen('TextStyle', window, 1);   % 0=normal text style. 1=bold. 2=italic.

end

if (params.is_sentence)
    WORD_DISPLAY_SEC       = WORD1_DISPLAY_SEC;
    POST_WORD_FIXATION_SEC = WHITE_DISPLAY_SEC;
end
WaitSecs('UntilTime', vbloff + INIT_FIXATION_DELAY_SEC - POST_WORD_FIXATION_SEC);
vbloff = vbloff + INIT_FIXATION_DELAY_SEC - POST_WORD_FIXATION_SEC;

% how the stimuli are grouped together:
if ((params.is_sentence) && (~params.stimuli_in_text_file))
    load(params.sentence_starts_fname, 'sentences_start');
end

ii_S = 1;
while (ii_S <= num_stimuli)

    if (~params.is_audio)
        if (~params.stimuli_in_text_file)
            Image = im_array{ii_S};
            winRect_stimuli = [0,0,1100,1100];
            %imageRect = DeterminImageSize(Image,winRect);
            imageRect = DeterminImageSize(Image,winRect_stimuli);
        
        %    imageScreenSize(ii_S,:) = imageRect;
        else
            cur_text = word_array{ii_S};
        end
        
        if (params.is_sentence)
            if (any(ii_S == sentences_start))
                WORD_DISPLAY_SEC = WORD1_DISPLAY_SEC;
            else
                WORD_DISPLAY_SEC = WORD2_DISPLAY_SEC;
            end
        end
        PresentTrainingStimuli;
        if (params.is_sentence)
            if (any(ii_S+1 == sentences_start))
                % next word starts a new sentence -- use inter-sentence delay:
                POST_WORD_FIXATION_SEC = POST_WORD2_FIXATION_SEC;
            else
                % intra-sentence delay:
                POST_WORD_FIXATION_SEC = WHITE_DISPLAY_SEC;
            end
        end
        
    else
        
        % audio:
        disp('playing now')
        send_ttl_now(SOUND_ON);
        [t1, t_end] = psych_play_sound_file(pahandle, audio_fname_cell, ii_S, ...
            t_end + POST_AUDIO_PLAYBACK_SEC);
        send_ttl_now(SOUND_OFF);
        key_presses_to_log();
        
    end
    
    if (is_exiting)
        break;
    end
    
%    if ~mod(ii_S,30)
        % Update main struct:
%        Stimuli_presentation.S_time_screen_on = S_time_screen_on;
%        Stimuli_presentation.user_answer_S = user_answer_S;
%        Stimuli_presentation.user_timing_S = user_timing_S;
%        Stimuli_presentation.imageScreenSize = imageScreenSize;
        
%        filename = fullfile(log_dir,sprintf('ONLINE_training_results_%f.mat',now));
%        save(filename,'Stimuli_presentation','ttl','TTL')
%    end
   
    ii_S = ii_S + 1;
end

% Update main struct:
%Stimuli_presentation.S_time_screen_on = S_time_screen_on;
%Stimuli_presentation.user_answer_S = user_answer_S;
%Stimuli_presentation.user_timing_S = user_timing_S;
%Stimuli_presentation.imageScreenSize = imageScreenSize;

%filename = fullfile(log_dir,sprintf('ONLINE_training_results_%f.mat',now));
%save(filename,'Stimuli_presentation','ttl','TTL','event_track')

% PsychPortAudio('Stop', pahandle(1));
% Screen('Close',imTextures);

if (params.is_audio)
    % Close the audio device:
    PsychPortAudio('Close', pahandle);
end

Screen(window,'Close');
%Screen('CloseAll');
%ShowCursor;
%Priority(0);
    

catch
    % catch error: This is executed in case something goes wrong in the
    % 'try' part due to programming error etc.:

    log_file_writer(GetSecs, sprintf('ERROR_CAUGHT EXITING'));

    % Do same cleanup as at the end of a regular session...
    session_cleanup;
    
    % Output the error message that describes the error:
    psychrethrow(psychlasterror);
end % try ... catch %
