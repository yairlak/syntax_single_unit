function [] = sound_ch_to_wav(ch)
% sound_ch_to_wav    

% Author: Ariel Tankus.
% Created: 28.06.2011.


wavefile = 'speech_session.wav';

if (nargin < 1)
    ch = 129;
end
load(sprintf('CSC%d.mat', ch));

sr = neuroport_samp_freq_hz;
%freq_low_hz  = 250;
%freq_high_hz = 3000;
freq_low_hz  = 60;
freq_high_hz = 4000;
[b, a] = ellip(2, 0.5, 20, [freq_low_hz, freq_high_hz]*2/sr);
%freq_hz = 60;
%freq_rel = freq_hz ./ sr;
%[b, a] = ellip(2, 0.5, 20, freq_hz*2/sr, 'high');
data_bandpass = filtfilt(b, a, double(data));

nbits = 16;
d = scl(data_bandpass, -1, 1);
%d = scl(data_bandpass, -10, 10);
%d((d > 1) | (d < -1)) = 1;
wavwrite(d, sr, nbits, wavefile);
