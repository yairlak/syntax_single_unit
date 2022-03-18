% DefTTLs() - script - to unify definitions between Train\Test functions
global TTL; TTL = 1;
global ttl; ttl = [];
global event_list;

ttlwait = 0.01; % Time between TTL pulses

%1 to 255 serial events (are global)
ii = 1; ttl_buf = 2;
global START_SEC; START_SEC = ii; ii = ii+ttl_buf;
global END_SEC;  END_SEC = ii; ii = ii+ttl_buf;

global IMAGE_ON; IMAGE_ON   = ii; ii = ii+ttl_buf;
global IMAGE_OFF; IMAGE_OFF = ii; ii = ii+ttl_buf;

global RES_UP; RES_UP = ii; ii = ii+ttl_buf;
global RES_DOWN; RES_DOWN = ii; ii = ii+ttl_buf;
global RES_LEFT; RES_LEFT = ii; ii = ii+ttl_buf;
global RES_RIGHT; RES_RIGHT = ii; ii = ii+ttl_buf;

global RES_SPACE; RES_SPACE = ii; ii = ii+ttl_buf;
global RES_CTRL; RES_CTRL = ii; ii = ii+ttl_buf;
global RES_ALT; RES_ALT = ii; ii = ii+ttl_buf;

global RES_1; RES_1 = ii; ii = ii+ttl_buf;
global RES_2; RES_2 = ii; ii = ii+ttl_buf;
global RES_3; RES_3 = ii; ii = ii+ttl_buf;
global RES_4; RES_4 = ii; ii = ii+ttl_buf;
global RES_5; RES_5 = ii; ii = ii+ttl_buf;
global RES_6; RES_6 = ii; ii = ii+ttl_buf;
global RES_7; RES_7 = ii; ii = ii+ttl_buf;
global RES_8; RES_8 = ii; ii = ii+ttl_buf;

global PRE_TRAIN_START; PRE_TRAIN_START = ii; ii = ii+ttl_buf;
global TRAIN_START; TRAIN_START = ii; ii = ii+ttl_buf;
global TEST_START; TEST_START = ii; ii = ii+ttl_buf;
global MAIN_PART_END; MAIN_PART_END = ii; ii = ii+ttl_buf;
global POST_STIM_TTL; POST_STIM_TTL = ii; ii = ii+ttl_buf;

global SOUND_ON; SOUND_ON = ii; ii = ii+ttl_buf;
global SOUND_OFF; SOUND_OFF = ii; ii = ii+ttl_buf;


% % event1 = 1; % image on
% % event3 = 3; % response - UpKey
% % event5 = 5; % response - DownKey
% % event8 = 8; % image off
% % event11 = 11; % start sub-section
% % event13 = 13; % end sub-section
% % event15 = 15; % response - LeftKey
% % event17 = 17; % response - RightKey
% % event19 = 19; % response - ControlKey
% % event21 = 21; % response - AltKey
% % event23 = 23; % response - space
eventreset = 0;     % Reset all bins to zero
event255 = 255;


event_list.START_SEC = START_SEC;
event_list.END_SEC = END_SEC;
event_list.IMAGE_ON = IMAGE_ON;
event_list.IMAGE_OFF = IMAGE_OFF;
event_list.RES_UP = RES_UP;
event_list.RES_DOWN = RES_DOWN;
event_list.RES_LEFT = RES_LEFT;
event_list.RES_RIGHT = RES_RIGHT;
event_list.RES_SPACE = RES_SPACE;
event_list.RES_CTRL = RES_CTRL;
event_list.RES_ALT = RES_ALT;
event_list.RES_1 = RES_1;
event_list.RES_2 = RES_2;
event_list.RES_3 = RES_3;
event_list.RES_4 = RES_4;
event_list.RES_5 = RES_5;
event_list.RES_6 = RES_6;
event_list.RES_7 = RES_7;
event_list.RES_8 = RES_8;
event_list.SOUND_ON = SOUND_ON;
event_list.SOUND_OFF = SOUND_OFF;
event_list.eventreset = eventreset;
event_list.event255 = event255;

