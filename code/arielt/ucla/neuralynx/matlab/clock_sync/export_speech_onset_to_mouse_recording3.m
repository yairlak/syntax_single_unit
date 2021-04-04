function export_speech_onset_to_mouse_recording3()
% export_speech_onset_to_mouse_recording3    

% Author: Ariel Tankus.
% Created: 16.08.2011.


if (~exist(['.', filesep, 'beep_events.log'], 'file'))
    [s, w] = unix('extract_beep_events');
    if (s ~= 0)
        error('Cannot run extract_beep_events: %s', w);
    end
end

[cue_time, ann] = textread('beep_events.log', '%d %s\n');
ann = rm_prefix_str_cell(ann, 'ANNOTATE_');

% BEEP_CTRL and IMAGERY have no speech onset:
has_behavior = ((~strcmp(ann, 'ANNOTATE_BEEP_CTRL')) & ...
                (cellfun('isempty', strfind(ann, 'ANNOTATE_IMAGERY'))));

speech_onset_time_sec = load('labels.txt');
speech_onset_time_sec = speech_onset_time_sec(:, 1);
speech_onset_time_usec = speech_onset_time_sec*1E6;
keyboard
export_annotations_to_mouse_recording2(speech_onset_time_sec, ann);
