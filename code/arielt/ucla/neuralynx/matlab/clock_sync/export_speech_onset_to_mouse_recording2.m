function export_speech_onset_to_mouse_recording2(rm_no_response_stimuli)
% export_speech_onset_to_mouse_recording2    

% Author: Ariel Tankus.
% Created: 30.06.2011.
% Modified: 25.09.2011.


if (nargin < 1)
    % remove all stimuli for which no overt user response is expected:
    rm_no_response_stimuli = true;
end

if (~exist(['.', filesep, 'beep_events_list.txt'], 'file'))
    [s, w] = unix('extract_beep_events');
    if (s ~= 0)
        error('Cannot run extract_beep_events: %s', w);
    end
end

ann = textread('beep_events_list.txt', '%s\n');
if (rm_no_response_stimuli)
    ann = ann(~strcmp(ann, 'ANNOTATE_BEEP_CTRL'));  % BEEP_CTRL has no speech onset.
    ann = ann(cellfun('isempty', strfind(ann, 'ANNOTATE_IMAGERY')));  % IMAGERY has no speech onset.
    ann = ann(cellfun('isempty', strfind(ann, 'ANNOTATE_IMAGINE')));  % IMAGERY has no speech onset.
end

speech_onset_time_sec = load('labels.txt');
speech_onset_time_sec = speech_onset_time_sec(:, 1);

export_annotations_to_mouse_recording2(speech_onset_time_sec, ann);
