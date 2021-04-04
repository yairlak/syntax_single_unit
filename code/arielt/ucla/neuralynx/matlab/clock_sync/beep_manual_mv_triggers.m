function new_start_inds = beep_manual_mv_triggers(start_inds, trigger_mv_tbl)
% beep_manual_mv_triggers    

% Author: Ariel Tankus.
% Created: 08.08.2010.


mv_offset_inds = round(trigger_mv_tbl(:,2)./1000.*neuroport_samp_freq_hz);

new_start_inds = start_inds;
for i=1:size(trigger_mv_tbl, 1)
    new_start_inds(trigger_mv_tbl(i, 1)) = ...
        start_inds(trigger_mv_tbl(i, 1)) + mv_offset_inds(i);
end
