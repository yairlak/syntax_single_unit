clear all; close all; clc
addpath('functions')

%%
patients = {'patient_480'};

%%  Main HTML for all patients
file_name = sprintf('HTML');

% open file
fileID = fopen(fullfile('..', '..', [file_name '.html']), 'w');

% Begining of file
fprintf(fileID, '<html>\n');
fprintf(fileID, '<head>\n');
fprintf(fileID, '<title>Rasters - all patients</title>\n');
fprintf(fileID, '</head>\n');
fprintf(fileID, '<body>\n');

%
for p = 1:length(patients)
    curr_patient = patients{p};
    fprintf(fileID, '<a href="main_rasters_syntax_patient_%s.html" title="Patient %s"> Patient %s</a><br><br>', curr_patient, curr_patient, curr_patient);
end
fclose(fileID);

%% Main HTML for Patient En_01
clear all; close all; clc

%  Generate HTML
file_name = sprintf('main_rasters_syntax_patient_En_01');
fileID = fopen(fullfile('..', '..', [file_name '.html']), 'w');

% Begining of file
fprintf(fileID, '<html>\n');
fprintf(fileID, '<head>\n');
fprintf(fileID, '<title>Rasters - En_01</title>\n');
fprintf(fileID, '</head>\n');
fprintf(fileID, '<body>\n');

%
fprintf(fileID, '<font>Patient En_01</font><br><br>\n');
% Sentences
fprintf(fileID, '<a href="rasters_syntax_En_01_sentences.html" title="Sentences"> Sentences</a><br><br>');
% Words
fprintf(fileID, '<a href="rasters_syntax_En_01_words.html" title="Words"> Words</a><br><br>');
% Words with key press
fprintf(fileID, '<a href="rasters_syntax_En_01_words_press.html" title="Words_press"> Words_press</a><br><br>');
% Nonwords
fprintf(fileID, '<a href="rasters_syntax_En_01_nonwords.html" title="Non    words"> Nonwords</a><br><br>');
% Key presses
fprintf(fileID, '<a href="syntax_single_unit_rasters_key_press_En_01.html" title="Key presses"> Key presses</a><br><br>');

fclose(fileID);

%% Main HTML for Patient ArM01
clear all; close all; clc
curr_patient = 'ArM01';
%  Generate HTML
file_name = sprintf('main_rasters_syntax_patient_ArM01');
fileID = fopen(fullfile('..', '..', [file_name '.html']), 'w');

% Begining of file
fprintf(fileID, '<html>\n');
fprintf(fileID, '<head>\n');
fprintf(fileID, '<title>Rasters - ArM01</title>\n');
fprintf(fileID, '</head>\n');
fprintf(fileID, '<body>\n');

%
fprintf(fileID, '<font>Patient ArM01</font><br><br>\n');
% Sentences
fprintf(fileID, '<a href="http://chechiklab.biu.ac.il/~yairlak/main_rasters_syntax_single_unit.html" title="Patient ArM01"> Sentences</a><br><br>');
% fprintf(fileID, '<a href="rasters_syntax_ArM01_sentences.html" title="Sentences"> Sentences</a><br><br>');
% Words
fprintf(fileID, '<a href="rasters_syntax_ArM01_words.html" title="Words"> Words</a><br><br>');
% Words with key press
fprintf(fileID, '<a href="rasters_syntax_ArM01_words_press.html" title="Words_press"> Words_press</a><br><br>');
% Nonwords
fprintf(fileID, '<a href="rasters_syntax_ArM01_nonwords.html" title="Nonwords"> Nonwords</a><br><br>');
% Nonwords with key press
fprintf(fileID, '<a href="rasters_syntax_ArM01_nonwords_press.html" title="Nonwords_press"> Nonwords_press</a><br><br>');
% Key presses
% fprintf(fileID, '<a href="syntax_single_unit_rasters_key_press_En_01.html" title="Key presses"> Key presses</a><br><br>');

fclose(fileID);

%%
patients = {'ArM01', 'En_01'};
for p = 1:length(patients)
    curr_patient = patients{p};

%% En_01 Sentences HTML
% clear all; close all;

%  Generate HTML
file_name = sprintf('rasters_syntax_%s_sentences', curr_patient);
fileID = fopen(fullfile('..', '..', [file_name '.html']), 'w');

% Begining of file
fprintf(fileID, '<html>\n');
fprintf(fileID, '<head>\n');
fprintf(fileID, '<title>Rasters - %s</title>\n', curr_patient);
fprintf(fileID, '</head>\n');
fprintf(fileID, '<body>\n');

fprintf(fileID, '<font>Patient %s Sentences</font><br><br>\n', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_sentences_all_trials.html" title="all_trials"> All trials</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_sentences_Sentence_type.html" title="Sentence_type"> Sentence_type</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_sentences_Sentence_type_four_words.html" title="Sentence_type_four_words"> Sentence_type_four_words</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_sentences_Subject_type.html" title="Subject_type"> Subject_type</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_sentences_Verb_type.html" title="Verb_type"> Verb_type</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_sentences_Subject_number.html" title="Subject_number"> Subject_number</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_sentences_Embedding.html" title="Embedding"> Embedding</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_sentences_Sentence_length.html" title="Sentence_length"> Sentence_length</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_sentences_Subject_gender.html" title="Subject_gender"> Subject_gender</a><br><br>', curr_patient);


%% Words HTML
% clear all; close all;

%  Generate HTML
file_name = sprintf('rasters_syntax_%s_words', curr_patient);
fileID = fopen(fullfile('..', '..', [file_name '.html']), 'w');

% Begining of file
fprintf(fileID, '<html>\n');
fprintf(fileID, '<head>\n');
fprintf(fileID, '<title>Rasters - %s</title>\n', curr_patient);
fprintf(fileID, '</head>\n');
fprintf(fileID, '<body>\n');

fprintf(fileID, '<font>Patient %s Words</font><br><br>\n', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_words_all_trials.html" title="all_trials">All trials</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_words_Num_of_Letters.html" title="Num_of_Letters">Num_of_letters</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_words_Num_of_Phonemes.html" title="Num_of_phonemes">Num_of_phonemes</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_words_Num_of_syllables.html" title="Num_of_syllables">Num_of_syllables</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_words_Num_of_affixes.html" title="Num_of_affixes">Num_of_affixes</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_words_Morphological_complexity.html" title="Morphological_complexity">Morphological_complexity</a><br><br>', curr_patient);

%% Words_press HTML
% clear all; close all;

%  Generate HTML
file_name = sprintf('rasters_syntax_%s_words_press', curr_patient);
fileID = fopen(fullfile('..', '..', [file_name '.html']), 'w');

% Begining of file
fprintf(fileID, '<html>\n');
fprintf(fileID, '<head>\n');
fprintf(fileID, '<title>Rasters - %s</title>\n', curr_patient);
fprintf(fileID, '</head>\n');
fprintf(fileID, '<body>\n');

fprintf(fileID, '<font>Patient %s Words_press</font><br><br>\n', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_words_press_all_trials.html" title="All trials">All trials</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_words_press_Num_of_Letters.html" title="Num_of_letters">Num_of_letters</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_words_press_Num_of_Phonemes.html" title="Num_of_phonemes">Num_of_phonemes</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_words_press_Num_of_syllables.html" title="Num_of_syllables">Num_of_syllables</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_words_press_Num_of_affixes.html" title="Num_of_affixes">Num_of_affixes</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_words_press_Morphological_complexity.html" title="Morphological_complexity">Morphological_complexity</a><br><br>', curr_patient);

%% Nonwords HTML
% clear all; close all;

%  Generate HTML
file_name = sprintf('rasters_syntax_%s_nonwords', curr_patient);
fileID = fopen(fullfile('..', '..', [file_name '.html']), 'w');

% Begining of file
fprintf(fileID, '<html>\n');
fprintf(fileID, '<head>\n');
fprintf(fileID, '<title>Rasters - %s</title>\n', curr_patient);
fprintf(fileID, '</head>\n');
fprintf(fileID, '<body>\n');

fprintf(fileID, '<font>Patient %s Nonwords</font><br><br>\n', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_nonwords_all_trials.html" title="all_trials">All trials</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_nonwords_Num_of_Letters.html" title="Num_of_letters">Num_of_letters</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_nonwords_Num_of_Phonemes.html" title="Num_of_phonemes">Num_of_phonemes</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_nonwords_Num_of_syllables.html" title="Num_of_syllables">Num_of_syllables</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_nonwords_Num_of_affixes.html" title="Num_of_affixes">Num_of_affixes</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_nonwords_Morphological_complexity.html" title="Morphological_complexity">Morphological_complexity</a><br><br>', curr_patient);

%% Nonwords press HTML
% clear all; close all;

%  Generate HTML
file_name = sprintf('rasters_syntax_%s_nonwords_press', curr_patient);
fileID = fopen(fullfile('..', '..', [file_name '.html']), 'w');

% Begining of file
fprintf(fileID, '<html>\n');
fprintf(fileID, '<head>\n');
fprintf(fileID, '<title>Rasters - %s</title>\n', curr_patient);
fprintf(fileID, '</head>\n');
fprintf(fileID, '<body>\n');

fprintf(fileID, '<font>Patient %s Nonwords_press</font><br><br>\n', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_nonwords_press_all_trials.html" title="all_trials">All trials</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_nonwords_press_Num_of_Letters.html" title="Num_of_letters">Num_of_letters</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_nonwords_press_Num_of_Phonemes.html" title="Num_of_phonemes">Num_of_phonemes</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_nonwords_press_Num_of_syllables.html" title="Num_of_syllables">Num_of_syllables</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_nonwords_press_Num_of_affixes.html" title="Num_of_affixes">Num_of_affixes</a><br><br>', curr_patient);
fprintf(fileID, '<a href="rasters_syntax_%s_nonwords_press_Morphological_complexity.html" title="Morphological_complexity">Morphological_complexity</a><br><br>', curr_patient);


end