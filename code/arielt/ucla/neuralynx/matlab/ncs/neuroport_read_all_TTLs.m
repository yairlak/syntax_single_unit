function [accumTimeStamps, accumTTL] = neuroport_read_all_TTLs(filename)
% neuroport_read_all_TTLs    

% Author: Ariel Tankus.
% Created: 26.01.2010.


if (isempty(findstr(filename,filesep)))
    filename = ['.', filesep, filename];   % filename must contain filesep for nevopen to operate correctly.
end
%r = nevopen(filename);
%[accumTimeStamps, accumTTL] = nevdigin;

output = openNEV(filename, 'read', 'nomat', 'nosave');
accumTimeStamps = output.Data.SerialDigitalIO.TimeStampSec .* 1E6;    % usec.

%accumTimeStamps = accumTimeStamps ./ neuroport_samp_freq_hz .* 1E6;    % usec.
accumTTL = double(output.Data.SerialDigitalIO.UnparsedData);
accumTTL = mod(accumTTL, 2^16);
%nevclose;
