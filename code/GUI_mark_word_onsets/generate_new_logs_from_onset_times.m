clear all; close all; clc

%% Which log to regenerate

path2logs = fullfile('..', '..', 'Data', 'UCLA', 'patient_513', 'Logs');
% log_file_name = 'events_log_in_cheetah_clock_block6.log'; %For EN_02, choose even numbers for AUDIO (part2, part4, part6)
log_file_name = 'events_log_in_cheetah_clock_part1.log';

%% Load onset-times file (same order as the sentences in the file below)
file_name = 'English_stimuli_word_onsets 1 to 152.txt';
fid = fopen(file_name, 'r');
onset_times = [];
while ~feof(fid)
    curr_line = fgetl(fid);
    curr_line = strsplit(curr_line, '\t');
    curr_line = cellfun(@str2double, curr_line);
    onset_times{curr_line(1)} = curr_line(2:end); % Starts from 2 since the first number is the sentence number 
end
sentence_length_according_to_onset_times = cellfun(@length, onset_times)';

%% Load sentnence (random) order sent to Ariel, which was used in the first block (same order as the onset-times file)
sentences_file_name = 'sentences_Eng_rand_En02.txt';
sentences = importdata(sentences_file_name);
sentences_words = cellfun(@strsplit, sentences, 'uniformoutput', false);
sentence_lengths = cellfun(@length, sentences_words);

if ~all(sentence_length_according_to_onset_times == sentence_lengths) % Sanity check
    error('Problem with onset times. Doesn''t match number of words in the sentence')
end

%% Read log file and generate a new one with more triggers for all words
conversion_factor = 1e6; % convert from sec (in onset_times array) to microsec (in log)
fid_old = fopen(fullfile(path2logs, log_file_name), 'r');
new_log_file_name = ['new_' log_file_name];
fid_new = fopen(fullfile(path2logs, new_log_file_name), 'w');
cnt_token = 1;
while ~feof(fid_old)
    curr_line = fgetl(fid_old);
    curr_fields = strsplit(curr_line);
    if strcmp(curr_fields{2}, 'AUDIO_PLAYBACK_ONSET')
       first_word_onset = curr_fields{1};
       IX = strfind(curr_fields{3}, '.wav');
       wav_file_number = curr_fields{3}(1:(IX-1));
       curr_onset_times = onset_times{str2double(wav_file_number)}; 
       
       curr_word = sentences_words{str2double(wav_file_number)}{1}; 
       new_line = sprintf('%s %s %i %s %i %s\n', curr_fields{1}, curr_fields{2}, cnt_token, wav_file_number, 1, curr_word);
       fprintf(fid_new, new_line); % Copy row of first-word onset.
       cnt_token = cnt_token + 1;
       for t = 1:length(curr_onset_times)
           new_onset_time = str2double(first_word_onset) + curr_onset_times(t)*conversion_factor; 
           if t < length(curr_onset_times)
               curr_word = sentences_words{str2double(wav_file_number)}{t+1}; 
               new_line = sprintf('%i %s %i %s %i %s\n', new_onset_time, curr_fields{2}, cnt_token, wav_file_number, t+1, curr_word);
               cnt_token = cnt_token + 1;
           else
               curr_word = 'END_OF_WAV'; 
               new_line = sprintf('%i %s %s %s %s\n', new_onset_time, curr_fields{2}, '_', wav_file_number, curr_word);
           end
           fprintf(fid_new, new_line); % Generate a new line of log for each word, together with its timing.           
       end
    else
         fprintf(fid_new, [curr_line '\n']);
    end
end
fclose('all');
