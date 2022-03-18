% Presenting image stimuli

ListenChar(2);
HideCursor();

if (~params.stimuli_in_text_file)
    tex = Screen('MakeTexture', window, Image);
    dstRect = CenterRectOnPoint(imageRect, 0.5*winRect(3), 0.5*windowRect(4));
    Screen('DrawTexture', window, tex, [], dstRect); % should now automatically scale your picture into the imsizepxl x imsizepxl box
else
    [nx, ny, textbounds] = DrawFormattedText(window, cur_text, 'center', 'center');
end

if (~exist('vbloff','var')) % Making sure there is a buffer between successive images
    vbloff = 0;
end

WaitSecs('UntilTime', vbloff + POST_WORD_FIXATION_SEC);

key_presses_to_log();
if (is_exiting)
    return;
end

vblon = Screen('Flip', window); % Update view --> image is on

send_ttl_now(IMAGE_ON);
if (~params.stimuli_in_text_file)
    log_file_writer(vblon, sprintf('DISPLAY_PICTURE %s', fname_array{ii_S}));
    %S_time_screen_on(ii_S) = vblon;
else
    log_file_writer(vblon, sprintf('DISPLAY_TEXT %d %d %d %s', ii_S, ...
                                   word_num_in_sentence(ii_S, 1), ...
                                   word_num_in_sentence(ii_S, 2), cur_text));
end

% FIXATION:
if ((~params.is_sentence) || (any(ii_S+1 == sentences_start)))
    % in sentences, use fixation only between sentences, not between words
    % within a sentence:
    tex_fixation = Screen('MakeTexture', window, im_fixation);
else
    tex_fixation = Screen('MakeTexture', window, white);
%    Screen('FillRect', window, [1, 1, 1]);  % white.
end
Screen('DrawTexture', window, tex_fixation);

WaitSecs('UntilTime', vblon + WORD_DISPLAY_SEC);

key_presses_to_log();
if (is_exiting)
    return;
end

%Screen('DrawTexture', window, blackscreen);             % Image off
vbloff = Screen('Flip', window); % Update view --> image is off
send_ttl_now(IMAGE_OFF);
if (~params.stimuli_in_text_file)
    log_file_writer(vbloff, sprintf('DISPLAY_PICTURE OFF'));
else
    log_file_writer(vbloff, sprintf('DISPLAY_TEXT OFF'));
end

%Screen('Close', tex);
FlushEvents('keyDown');
ListenChar(0);
