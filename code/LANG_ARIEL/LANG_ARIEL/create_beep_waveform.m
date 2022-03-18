function y = create_beep_waveform(freq_hz, rec_len_sec, ...
                                  sampling_freq_hz)
% create_beep_waveform    

% Author: Ariel Tankus.
% Created: 25.11.2015.


if (nargin < 1)
    freq_hz = 400;     % sound freq_hzuency [Hz]
end
if (nargin < 2)
    rec_len_sec = 1;    % 1 sec. wav file.
end
if (nargin < 3)
    sampling_freq_hz = 8000;      % #observations per second in the wav file.
end

x = (1/sampling_freq_hz):(1/sampling_freq_hz):rec_len_sec;
y = sin(x.*2.*pi.*freq_hz);
