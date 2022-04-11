function LANG_SENTENCES_TEXT(is_rand)
% LANG_SENTENCES_TEXT    

% Author: Ariel Tankus.
% Created: 17.03.2017.


if ((nargin < 1) || (is_rand == false))
    filename_stimuli = 'params_sentences_text';
else
    filename_stimuli = 'params_sentences_text_rand';
end
lang_paradigm(filename_stimuli);
