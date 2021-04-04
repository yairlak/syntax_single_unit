function [accumTimeStamps, accumTTL] = neuralynx_read_all_TTLs(filename)
% neuralynx_read_all_TTLs    
% [uptimes, downtimes]=load_nev(filename)
% This function reads in the specified Neuralynx .Nev file, and outputs the
% upstrokes and downstrokes from it.
%
% From our setup at UCLA, we found that the TTL value of -2 corresponds to an
% upstroke, and -4 corresponds to a downstroke.  This function makes that assumption!

% Author: Ariel Tankus.
% Created: 26.01.2010.


accumTimeStamps = [];
accumTTL = [];

%upstrokes=[];
%downstrokes=[];

fid = fopen(filename,'r','l');

fseek(fid,16*1024,'bof');
    
while ~feof(fid)
    
  pktStart=fread(fid,1,'int16');
  pktId=fread(fid,1,'int16');
  pktDataSize=fread(fid,1,'int16');
  timeStamp=fread(fid,1,'int64');
  eventId=fread(fid,1,'int16');
  ttlValue=fread(fid,1,'uint16');
  crc=fread(fid,1,'int16');
  dummy=fread(fid,1,'int32');
  extra=fread(fid,8,'int32');
  eventString=char(fread(fid,128,'char')');

  accumTimeStamps = [accumTimeStamps; timeStamp];
  accumTTL = [accumTTL; ttlValue];
  
%  if ((ttlValue== -2) | (ttlValue== -3) | (ttlValue == -5))
%    upstrokes=[upstrokes timeStamp];
%  elseif ((ttlValue==-4) | (ttlValue==-1) | (ttlValue == -13))
%    downstrokes=[downstrokes timeStamp];
%  end

end
fclose(fid);
