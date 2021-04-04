function [] = cmp_beep_label()
% cmp_beep_label    

% Author: Ariel Tankus.
% Created: 19.12.2011.


load(['.', filesep, 'beep_event_times.log']);
load(['.', filesep, 'labels.txt']);

ann = textread(['.', filesep, 'beep_events_list.txt'], '%s\n');
ann_inds = ((~strcmp(ann, 'ANNOTATE_BEEP_CTRL')) & ... % BEEP_CTRL has no speech onset.
            (cellfun('isempty', strfind(ann, 'ANNOTATE_IMAGERY'))) & ...  % IMAGERY has no speech onset.
            (cellfun('isempty', strfind(ann, 'ANNOTATE_IMAGINE'))));  % IMAGINE has no speech onset.

% remove irrelevant events from list:
beep_event_times = beep_event_times(ann_inds);

len_beeps  = length(beep_event_times);
len_labels = size(labels, 1);
min_len = min(len_beeps, len_labels);

if (len_beeps ~= len_labels)
    fprintf('Different #events: beeps: %d;   labels: %d.', len_beeps, len_labels);
end
d = labels(1:min_len, 1) - beep_event_times(1:min_len)./1E6;
r = range2(d);
if (r(1) < 0)
    fprintf('Onset before cue:\n');
    fprintf('ind =');
    fprintf(' %d', find(d < 0));
    fprintf('\n');
end
if (r(2) > 1.5)
    fprintf('Onset later than 1.5s after cue:');
    fprintf('ind =');
    ind = find(d > 1.5);
    fprintf('%d.  %d - %d = %d', [ind, labels(ind, 1), ...
                        beep_event_times(ind)./1E6, d(ind)]');
    fprintf('\n');
end
