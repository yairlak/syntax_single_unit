function [] = create_params_file()
% create_params_file    

% Author: Ariel Tankus.
% Created: 31.01.2017.


%l = dir('Stimuli/Figures_murkamot/*.png');
%params.stimuliList = {l.name};

params.loading_slide = 'Stimuli/loading_slide.png';
params.fixation      = 'Stimuli/fixation.png';
params.is_sentence = false;
params.is_audio    = false;
params.stimuli_in_text_file = false;
params.use_rand_perm        = false;

params.instructions   = 'Stimuli/instructions_murkamot.png';
params.stimuli_subdir = 'Stimuli/Figures_murkamot';
params.stimuliList = {};
for i=1:295
    params.stimuliList = [params.stimuliList, {sprintf('%d.png', i)}];
end

save params_murkamot.mat params;

params.instructions   = 'Stimuli/instructions_nonwords.png';
params.stimuli_subdir = 'Stimuli/Figures_nonwords';
params.stimuliList = {};
for i=1:76
    params.stimuliList = [params.stimuliList, {sprintf('%d.png', i)}];
end

save params_nonwords.mat params;

params.instructions   = 'Stimuli/instructions_murkamot_press.png';
params.stimuli_subdir = 'Stimuli/Figures_murkamot';
params.stimuliList = {};
for i=1:295
    params.stimuliList = [params.stimuliList, {sprintf('%d.png', i)}];
end

save params_murkamot_press.mat params;

params.is_sentence           = true;
params.sentence_starts_fname = 'Stimuli/sentences_start.mat';
params.instructions          = 'Stimuli/instructions_sentences.png';
params.stimuli_subdir        = 'Stimuli/Figures_sentences';
params.stimuliList = {};
for i=1:702
    params.stimuliList = [params.stimuliList, {sprintf('%d.png', i)}];
end

save params_sentences.mat params;

params.is_audio       = true;
params.is_sentence    = false;
params.instructions   = 'Stimuli/instructions_audio_sentences.png';
params.stimuli_subdir = 'Stimuli/Sound_sentences';
params.stimuliList = {};
for i=1:152
    params.stimuliList = [params.stimuliList, {sprintf('%d.wav', i)}];
end

save params_audio_sentences0.mat params;

num_permutes = 6;
for i=1:(num_permutes-1)
    params.stimuliList = params.stimuliList(randperm(length(params.stimuliList)));

    save(sprintf('params_audio_sentences%d.mat', i), 'params');
end

%%%%%%%%%%%


params.is_sentence           = true;
params.is_audio              = false;
params.stimuli_in_text_file  = true;
params.use_rand_perm         = false;
params.sentence_starts_fname = 'Stimuli/sentences_start.mat';
params.instructions          = 'Stimuli/instructions_sentences.png';
params.stimuli_subdir        = 'Stimuli';
params.stimuliList           = 'sentences_Eng_rand_En02.txt';

save params_sentences_text.mat params;

params.use_rand_perm         = true;

save params_sentences_text_rand.mat params;
