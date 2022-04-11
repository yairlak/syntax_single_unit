% Presenting explanation slide - response is space
ListenChar(0);

defKeyboard;

HideCursor();

% Explanation slides have a fixed size
[M,N,~] = size(Image);
imageRect = [0 0 N M]; % we want all images to show up imsizepxl x imsizepxl pixels
dstRect = CenterRect(imageRect, winRect);

tex = Screen('MakeTexture', window, Image);
Screen('DrawTexture', window, tex, [], dstRect); % should now automatically scale your picture into the imsizepxl x imsizepxl box
vblon = Screen('Flip', window); % Update view --> image is on
time_screen_on = GetSecs;

send_ttl_now(IMAGE_ON);
log_file_writer(vblon, sprintf('DISPLAY_INSTRUCTIONS %s', instructions_fname));

wait_for_key(spaceKey);

% KbWait([],2); % make sure key is released
send_ttl_now(RES_SPACE);


% FIXATION:
im_loading = imread(params.loading_slide);   % Load "Loading..." image
tex_fixation = Screen('MakeTexture', window, im_loading);
Screen('DrawTexture', window, tex_fixation);
vbloff = Screen('Flip', window); % Update view --> image is off

send_ttl_now(IMAGE_OFF);
log_file_writer(vbloff, 'DISPLAY_INSTRUCTIONS OFF');

% psych_play_seq_sounds();

FlushEvents('keyDown');
ListenChar(0);
