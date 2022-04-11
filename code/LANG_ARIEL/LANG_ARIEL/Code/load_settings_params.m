function [settings, params] = load_settings_params()
%% Paths
settings.path2sentences = fullfile('..', 'Stimuli', 'Figures_sentences');
settings.path2murkamot = fullfile('..', 'Stimuli', 'Figures_murkamot');
settings.path2nonwords = fullfile('..', 'Stimuli', 'Figures_nonwords');

%% General params 
params.num_sentences = 200;  
params.num_murkamot = 295;  
params.num_nonwords = 76;  

%% Time durations
params.word_duration = 0.2; %sec
params.between_sentences = 2; %sec
params.between_words = 0.3; %sec - only for sentences block

end
