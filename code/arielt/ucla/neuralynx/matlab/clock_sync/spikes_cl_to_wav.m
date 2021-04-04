function [] = spikes_cl_to_wav(cl)
% spikes_cl_to_wav    

% Author: Ariel Tankus.
% Created: 05.01.2012.


wavefile = sprintf('speech_spikes%d.wav', cl);

% duration of sound for 1spike:
sound_duration_ms = 10;    % millisec.
sound_duration_micsec = sound_duration_ms.*1000;    % microsec.
sound_freq_hz = 200;

load(sprintf('CSC%d_cluster.mat', cl));

sr = neuroport_samp_freq_hz;
sr_microsec = 1E6/sr;
data = zeros(1, round(timeend./sr_microsec));

spike_times_microsec = 1E6.*spike_times_sec;

dt = round(sr/1000.*sound_duration_ms);    % #samples during sound for 1 spike.
inds_for_1cycle = (0:dt)./dt*2*pi;
num_cycles = sound_freq_hz./1000.*sound_duration_ms;
inds_for_all_cycles = inds_for_1cycle.*num_cycles;

for i=1:length(spike_times_microsec)
    t = round(spike_times_microsec(i)./1E6.*sr);
    data(t:(t+dt)) = sin(inds_for_all_cycles);
end

nbits = 16;
d = scl(data, -1, 1);
%d = scl(data_bandpass, -10, 10);
%d((d > 1) | (d < -1)) = 1;
wavwrite(d, sr, nbits, wavefile);
