% nKeys = inputdlg('number of gamepad key presses to check?\n');
% Make sure MAC restarted and MATLAB opened AFTER connecting the USB cable
% (to daq)
location='TLVMC';  %options: 'UCLA' or 'TLVMC', affecting hardware to use for TTL
DefTTLs() % Define key-presses, TTL events
defKeyboard;

dio = 3;
global portA; portA = 0;
global portB; portB = 1;

% dio=DaqDeviceIndex;                                     	% get a handle for the USB-1208FS
hwline=DaqDConfigPort(dio,0,0);                            	% configure digital port A for output
DaqDOut(dio,0,eventreset);
hwline=DaqDConfigPort(dio,1,0);                          	% configure digital port B for output
DaqDOut(dio,1,eventreset); laststim = 0;

ttlwait     = 0.001;

for i = 1:3
    DaqDOut(dio,portA,1); WaitSecs(ttlwait);DaqDOut(dio,portA,eventreset); WaitSecs(0.05);
    DaqDOut(dio,portA,2); WaitSecs(ttlwait);DaqDOut(dio,portA,eventreset); WaitSecs(0.05);
    DaqDOut(dio,portA,4); WaitSecs(ttlwait);DaqDOut(dio,portA,eventreset); WaitSecs(0.05);
    DaqDOut(dio,portA,8); WaitSecs(ttlwait);DaqDOut(dio,portA,eventreset); WaitSecs(0.05);
    DaqDOut(dio,portA,16); WaitSecs(ttlwait);DaqDOut(dio,portA,eventreset); WaitSecs(0.05);
    DaqDOut(dio,portA,32); WaitSecs(ttlwait);DaqDOut(dio,portA,eventreset); WaitSecs(0.05);
    DaqDOut(dio,portA,64); WaitSecs(ttlwait);DaqDOut(dio,portA,eventreset); WaitSecs(0.05);
    DaqDOut(dio,portA,128); WaitSecs(ttlwait);DaqDOut(dio,portA,eventreset); WaitSecs(0.05);
    DaqDOut(dio,portA,255); WaitSecs(ttlwait);DaqDOut(dio,portA,eventreset); WaitSecs(0.3);
end



ListenChar(2);

disp('Press space')
% Wait for reaction
CheckKeyPress(dio,spaceKey,RES_SPACE,location)
WaitSecs(0.3);

disp('Press ctrl')
CheckKeyPress(dio,controlKey,RES_CTRL,location)
WaitSecs(0.3);

disp('Press alt')
CheckKeyPress(dio,altKey,RES_ALT,location)
WaitSecs(0.3);

disp('Press up')
CheckKeyPress(dio,upKey,RES_UP,location)
WaitSecs(0.3);

disp('Press right')
CheckKeyPress(dio,rightKey,RES_RIGHT,location)
WaitSecs(0.3);

disp('Press left')
CheckKeyPress(dio,leftKey,RES_LEFT,location)
WaitSecs(0.3);

disp('Press down')
CheckKeyPress(dio,downKey,RES_DOWN,location)
WaitSecs(0.3);


WaitSecs(0.5);

if strcmp(location,'UCLA') % mark the end of the experiment with 3 255 triggers separated 100ms from each other
    for i=1:3
        DaqDOut(dio,portA,event255); % send ?eventX' TTL (0-255)
        WaitSecs(ttlwait);
        DaqDOut(dio,portA,eventreset); % reset Daq interface
        WaitSecs(0.1);
    end
end


ListenChar(0);
