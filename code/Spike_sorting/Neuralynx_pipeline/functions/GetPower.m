function [f, amp] = GetPower(signal, SamplingRate)

Y = fft(signal);
Pyy = Y.* conj(Y) / length(signal);
p_spec = Pyy ;
amp = p_spec(1:ceil(length(p_spec)/2)) ;
f = SamplingRate*(1:length(amp))/(length(amp)*2);
amp = sqrt(amp(1:length(f)));
