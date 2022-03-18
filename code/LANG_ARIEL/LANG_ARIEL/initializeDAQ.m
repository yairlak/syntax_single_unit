function dio = initializeDAQ
% ttlon = 1;
ttloff = 0;
dio=DaqDeviceIndex;                 % get a handle for the USB-1208FS
hwline=DaqDConfigPort(dio,0,0);     % configure digital port A for output
DaqDOut(dio,0,ttloff);
hwline=DaqDConfigPort(dio,1,0);     % configure digital port B for output
DaqDOut(dio,1,ttloff);
% laststim = 0;