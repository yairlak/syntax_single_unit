function export_speech_onset_to_mouse_recording(beep_seq_len)
% export_speech_onset_to_mouse_recording    

% Author: Ariel Tankus.
% Created: 28.06.2011.


if (nargin < 1)
    beep_seq_len = 8;
end

if (~exist(['.', filesep, '/used_ann_list.txt'], 'file'))
    [s, w] = unix('extract_used_ann mouse_recording.log');
    if (s ~= 0)
        error('extract_used_ann mouse_recording.log failed: %s', w);
    end
end

ann = textread('used_ann_list.txt', '%s\n');
ann = ann(~strcmp(ann, 'ANNOTATE_BEEP_CTRL'));  % BEEP_CTRL has no speech onset.

speech_onset_time_sec = load('labels.txt');
speech_onset_time_sec = speech_onset_time_sec(:, 1);

export_annotations_to_mouse_recording(speech_onset_time_sec, ann, beep_seq_len);
