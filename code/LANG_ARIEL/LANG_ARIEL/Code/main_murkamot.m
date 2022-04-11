function main_murkamot(subject,run)
% Add_Psych;
%% Initialize
warning off
% HideCursor

% FOR DEBUG ONLY (COMMENT OUT IF NOT DEBUGGING)
PsychDebugWindowConfiguration([0],[0.5])
% --------------------------------------------

rect = get(0, 'ScreenSize');
rect = [0 0 rect(3:4)];
win = Screen('OpenWindow',0,[0 0 0],rect);
% win = Screen('OpenWindow', 0, [0 0 0], [50 50 640 480]);

[settings, params] = load_settings_params();
for i = 1:params.num_murkamot
    stimuli(i).image = imread(fullfile(settings.path2murkamot, sprintf('%i.png', i)));
    stimuli(i).image = Screen('MakeTexture', win, stimuli(i).image);
end

Screen(win,'TextSize',30);
Screen('DrawText',win,'we will start in a few seconds', rect(3)/3, rect(4)/2,[255 255 255]);
Screen('Flip',win);

%% Begin experiment
middleKey = KbName('m');
subject_responses = cell(params.num_murkamot, 1);

for i = 1:params.num_murkamot
    %% Fixation
    trial = tic;
    Screen('DrawText',win,'*', rect(3)/2, rect(4)/2,[255 255 255]);
    Screen('Flip',win);
    while toc < params.between_sentences % Wait until fixation time is over
        [~, ~, keyCode] = KbCheck; % Listen to the keyboard for subjects response
        if keyCode(middleKey)
            subject_responses{i, 1} = 'm'; 
            subject_responses{i, 2} = toc(rt);
        elseif keyCode(escKey)
            DisableKeysForKbCheck([]);
            Screen('CloseAll');
            return
        end
    end    
   
    %% Present visual stimulus
    Screen('DrawTexture', win, stimuli(i).image);
    Screen('Flip',win); 
    rt = tic;
    %% Wait for response
    while toc(trial) < params.word_duration
    end
    
end

%%save data
toc(grandStart)
DisableKeysForKbCheck([]);
Screen('CloseAll');
filename = sprintf('Subject_%s_run_%i_murkamot', subject, run);
save(fullfile('..', 'Output', filename))
% Remove_Psych;
end