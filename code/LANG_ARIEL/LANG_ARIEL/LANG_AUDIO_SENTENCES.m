function LANG_AUDIO_SENTENCES(perm_num)
% LANG_AUDIO_SENTENCES    
%
%                         LANG_AUDIO_SENTENCES(perm_num)
%                         perm_num - [0..5] - serial number of the
%                                             permutation of stimuli to use.

% Author: Ariel Tankus.
% Created: 16.03.2017.


if (nargin < 1)
    perm_num = 0;
end

filename_stimuli = sprintf('params_audio_sentences%d', perm_num);
lang_paradigm(filename_stimuli);
