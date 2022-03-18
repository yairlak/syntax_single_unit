function [] = create_dummy_beeps()
% create_dummy_beeps    

% Author: Ariel Tankus.
% Created: 16.03.2017.


cur_wd = pwd;
cd /media/arielt/ariel1/arielt/lang_project/from_maya/LANG_ARIEL/Stimuli/dummy_beeps;

num_freqs = 14;
num_lens = 5;

Fs = 44100;
rec_len = 0.1;
freq = 350;

delta_freq = 50; 
delta_rec_len = 0.3;    % sec.

for j=1:num_freqs
    for i=1:num_lens
        create_beep_wav(freq + (j-1).*delta_freq, ...
                        rec_len + (i-1).*delta_rec_len, Fs, false);
    end
end

cd(cur_wd);
