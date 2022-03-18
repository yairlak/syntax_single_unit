function s = initializeSerialPort

if(strfind(computer,'MAC'))
    try
        IOPort('CloseAll');WaitSecs(0.2); IOPort('CloseAll'); WaitSecs(0.2);
        [s err]=IOPort('OpenSerialPort','/dev/cu.usbserial','BaudRate=115200');
        
        if(isempty(err))
            disp('Serial initialization success');
        else
            error(err);
        end
        WaitSecs(0.4);
    catch ME
        rethrow(ME);
        warndlg('Error setting up triggers! Try restarting MATLAB with everything plugged in');
        %display('check that you have drivers for the usb-serial port cable. Check that psychtoolbox is installed.');
        return
    end
elseif strfind(computer,'WIN')
    disp('initializing serial port...')      % display status
    s = serial('COM3','BaudRate',115200,'Terminator',[]); % set serial port at baud rate of neuroport
    fopen(s);                                    % open serial port
    val = 'serial_init'; fprintf(s,char(val)); pause(0.01);
    
end