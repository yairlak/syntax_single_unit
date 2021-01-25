function export_beep_ttls_to_mouse_recording(start_inds)
% export_beep_ttls_to_mouse_recording    

% Author: Ariel Tankus.
% Created: 10.08.2010.


ann = textread('/home/arielt/ucla/center_out_joy2/motor_screening/annotation_headers_speech_screening.txt', '%s\n');
beep_seq_len = 6;

ttl_times_sec = start_inds ./ neuroport_samp_freq_hz;
ttl_times_sec_round = floor(ttl_times_sec);
ttl_times_usec = round((ttl_times_sec - ttl_times_sec_round) * 1E6);

fid = fopen('/tmp/mouse_recording_in_cheetah_clock_from_beeps.log', 'w');
if (fid == -1)
    error('Cannot open log file /tmp/mouse_recording_from_beeps.log');
end

cur_ann_counter = 0;
for i=1:length(ttl_times_sec_round)
    prev_usec = ttl_times_usec(i) - 1;
    if (prev_usec < 0)
        prev_usec = 1E6 + prev_usec;
        prev_sec = ttl_times_sec_round(i) - 1;
    else
        prev_sec = ttl_times_sec_round(i);
    end
    if (rem(i, beep_seq_len) == 1)
        % print annotation header:
        prev_prev_usec = prev_usec - 1;
        if (prev_prev_usec < 0)
            prev_prev_usec = 1E6 + prev_prev_usec;
            prev_prev_sec = prev_sec - 1;
        else
            prev_prev_sec = prev_sec;
        end
        cur_ann_counter = cur_ann_counter + 1;
        fprintf(fid, '%d%06d Event: %s START\n', prev_prev_sec, ...
                prev_prev_usec, ann{cur_ann_counter});
    end
    fprintf(fid, '%d%06d Event: KEY_PRESS 0 Go\n', prev_sec, prev_usec);
    fprintf(fid, '%d%06d Event: AUDIO_START 1\n', ttl_times_sec_round(i), ...
            ttl_times_usec(i));
    if (rem(i, beep_seq_len) == 0)
        next_usec = ttl_times_usec(i) + 1;
        if (next_usec > 1E6)
            next_usec = next_usec - 1E6;
            next_sec = ttl_times_sec_round(i) + 1;
        else
            next_sec = ttl_times_sec_round(i);
        end
        fprintf(fid, '%d%06d Event: %s END\n', next_sec, next_usec, ...
                ann{cur_ann_counter});
    end
end

fclose(fid);
