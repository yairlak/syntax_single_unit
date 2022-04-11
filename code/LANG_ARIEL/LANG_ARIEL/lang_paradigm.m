  % Running memory-location pairing - UCLA - in the cnl3 laptop
function lang_paradigm(filename_stimuli)

%%          ~~~~~~~~~~~~ FIND   RELATIVE PATH  ~~~~~~~~~~~~

if (nargin < 1)
    filename_stimuli = './murkamot_params';
end
    
close all;
% PsychJavaTrouble()

pathtoParadigmFolder = mfilename('fullpath'); % looks like this: /Users/NVT/OBJREC_FREERECALL_SOUND/objrec_adaptive_
here = strfind(pathtoParadigmFolder,filesep);  
pathtoParadigmFolder=[pathtoParadigmFolder(1:here(end)-1)];
%addpath(genpath(fullfile(pathtoParadigmFolder  )));
cd(pathtoParadigmFolder)
diary(fullfile(pathtoParadigmFolder,'LOGS',sprintf('command_window_log_MemParadigm_%f',now)))

log_basename = 'log_patient/events_log';

INIT_FIXATION_DELAY_SEC  =  5.000;
%WORD_DISPLAY_SEC         =  0.300;
%POST_WORD_FIXATION_SEC   =  1.700;
WORD_DISPLAY_SEC         =  0.200;
POST_WORD_FIXATION_SEC   =  1.800;

%WORD1_DISPLAY_SEC        =  0.300;
%WHITE_DISPLAY_SEC        =  0.050;
%WORD2_DISPLAY_SEC        =  0.300;
WORD1_DISPLAY_SEC        =  0.200;
WHITE_DISPLAY_SEC        =  0.300;
WORD2_DISPLAY_SEC        =  0.200;
POST_WORD2_FIXATION_SEC  =  1;

POST_AUDIO_PLAYBACK_SEC  =  1;

%% Make sure TTLs look good before starting actual recording
% testTTLs()
% /Users/mayags/Documents/LA_paradigms/MemLocPairing/CheckImageFiles.m
%global event_track; event_track = [];
global sendTTL;
global debug; 
if isempty(debug)
debug = 0;
end
global dio;
global serial_port;
global ttlLog;
global is_exiting;

is_exiting = false;
ttlLog = {};

%#################################################################
%   Send TTLs though the DAQ hardware interface
sendTTL = questdlg('Send TTLs ???','TTLs status','Yes (recording session)','No (just playing)','Yes (recording session)');
if sendTTL(1) == 'Y', sendTTL = 1; else sendTTL = 0; end
if (~sendTTL)
    h_msgbox = msgbox('TTLs  will  *NOT*  be  sent - are you sure you want to continue?','TTLs','modal');
    uiwait(h_msgbox);
end
%if ~sendTTL, warndlg('TTLs  will  *NOT*  be  sent - are you sure you want to continue?','TTLs','modal'); end
%################################################################

log_file_writer('open', log_basename);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%UCLA TTL settings
global location; location='UCLA';  %options: 'UCLA' or 'TLVMC', affecting hardware to use for TTL
global portA; portA = 0;
global portB; portB = 1;
% Screen('Preference', 'SkipSyncTests', 1);
dio = 3;

if debug
    dbstop if error;
    Screen('Preference','VisualDebugLevel',1);

Screen('Preference','SkipSyncTests',1);

    PsychDebugWindowConfiguration;
    % else
    %     HideCursor;
    %     ListenChar(2);
end

DefTTLs()

%% Initialising TTL hardware
if sendTTL && strcmp(location,'TLVMC')
%    params.sio = '/dev/tty.usbserial';
    params.sio = '/dev/ttyUSB0';
    % params.sio = '/dev/cu.usbserial'; % UCLA serial port
    sio = serial(params.sio,'BaudRate',115200,'Terminator', []);
    fopen(sio);     %remember to close manually if you terminal execution early
    fwrite(sio,1); WaitSecs(ttlwait); fwrite(sio,eventreset); WaitSecs(0.5);
    fwrite(sio,2); WaitSecs(ttlwait); fwrite(sio,eventreset); WaitSecs(0.5);
    fwrite(sio,3); WaitSecs(ttlwait); fwrite(sio,eventreset); WaitSecs(0.5);
    fwrite(sio,4); WaitSecs(ttlwait); fwrite(sio,eventreset);
end

if sendTTL && strcmp(location,'TLVMC') % mark the beginning of the experiment with four 255 triggers separated 100ms from each other
    fwrite(sio,event255); WaitSecs(ttlwait); fwrite(sio,eventreset); WaitSecs(0.1);
    fwrite(sio,event255); WaitSecs(ttlwait); fwrite(sio,eventreset); WaitSecs(0.1);
    fwrite(sio,event255); WaitSecs(ttlwait); fwrite(sio,eventreset); WaitSecs(0.1);
    fwrite(sio,event255); WaitSecs(ttlwait); fwrite(sio,eventreset);
end

%% Initialize DAQ
if sendTTL && strcmp(location,'UCLA')
    dio = initializeDAQ;
    serial_port = [];%initializeSerialPort;
%    % dio = DaqDeviceIndex;                                     	 % get a handle for the USB-1208FS
%    hwline = DaqDConfigPort(dio,0,0);                                % configure digital port A for output
%    DaqDOut(dio,portA,eventreset);
%    hwline = DaqDConfigPort(dio,1,0);              	              % configure digital port B for output
%    DaqDOut(dio,portB,eventreset); laststim = 0;
end

send_ttl_now(START_SEC);
%event_track{TTL} = sprintf('Starting main section 0');
%TTL = TTL + 1;

%initialize_TTLs()sca
;

stimuli_subdir = 'Stimuli';
  PsychToolbox_PresentPics_Training

session_cleanup();


end % wrapper func.


function initialize_TTLs()

global event_track;
global sendTTL;
global debug;


%UCLA TTL settings
location='UCLA';  %options: 'UCLA' or 'TLVMC', affecting hardware to use for TTL
global portA; portA = 0;
global portB; portB = 1;
% Screen('Preference', 'SkipSyncTests', 1);
dio = 3;

if debug
    dbstop if error;
    % else
    %     HideCursor;
    %     ListenChar(2);
end

DefTTLs();

%% Initialising TTL hardware
if sendTTL && strcmp(location,'TLVMC')
    params.sio = '/dev/tty.usbserial';
    % params.sio = '/dev/cu.usbserial';
    sio = serial(params.sio,'BaudRate',115200,'Terminator', []);
    fopen(sio);     %remember to close manually if you terminal execution early
    fwrite(sio,1); WaitSecs(ttlwait); fwrite(sio,eventreset); WaitSecs(0.5);
    fwrite(sio,2); WaitSecs(ttlwait); fwrite(sio,eventreset); WaitSecs(0.5);
    fwrite(sio,3); WaitSecs(ttlwait); fwrite(sio,eventreset); WaitSecs(0.5);
    fwrite(sio,4); WaitSecs(ttlwait); fwrite(sio,eventreset);
end

if sendTTL && strcmp(location,'TLVMC') % mark the beginning of the experiment with four 255 triggers separated 100ms from each other
    fwrite(sio,event255); WaitSecs(ttlwait); fwrite(sio,eventreset); WaitSecs(0.1);
    fwrite(sio,event255); WaitSecs(ttlwait); fwrite(sio,eventreset); WaitSecs(0.1);
    fwrite(sio,event255); WaitSecs(ttlwait); fwrite(sio,eventreset); WaitSecs(0.1);
    fwrite(sio,event255); WaitSecs(ttlwait); fwrite(sio,eventreset);
end

%% Initialize DAQ
if sendTTL && strcmp(location,'UCLA')
    % dio = DaqDeviceIndex;                                     	              % get a handle for the USB-1208FS
    hwline = DaqDConfigPort(dio,0,0);                                         % configure digital port A for output
    DaqDOut(dio,portA,eventreset);
    hwline = DaqDConfigPort(dio,1,0);                          	              % configure digital port B for output
    DaqDOut(dio,portB,eventreset); laststim = 0;
end

if sendTTL && strcmp(location,'UCLA') % mark the beginning of the experiment with four 255 triggers separated 100ms from each other
    for i=1:4
        DaqDOut(dio,portA,event255); % send ?eventX' TTL (0-255)
        WaitSecs(ttlwait);
        DaqDOut(dio,portA,eventreset); % reset Daq interface
        WaitSecs(0.1);
    end
end

if sendTTL && strcmp(location,'UCLA'); DaqDOut(dio,portA,eventname); WaitSecs(ttlwait); DaqDOut(dio,portA,eventreset); ttl(TTL,1) = GetSecs; ttl(TTL,2) = eventname;  % send a TTL to signal section start
elseif sendTTL && strcmp(location,'TLVMC'); fwrite(sio, eventname); ttl(TTL,1) = GetSecs; ttl(TTL,2) = START_SEC;
end
event_track{TTL} = sprintf('Starting main section 0');
TTL = TTL + 1;

end
