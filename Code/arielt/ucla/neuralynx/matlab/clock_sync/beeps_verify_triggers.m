function [] = beeps_verify_triggers()
% beeps_verify_triggers    

% Author: Ariel Tankus.
% Created: 08.08.2010.


% Manual verification of triggers by playing the relevant parts:
load trigger_start_inds;
load CSC129.mat;

sr = neuroport_samp_freq_hz;
freq_low_hz  = 60;
freq_high_hz = 4000;
[b, a] = ellip(2, 0.5, 20, [freq_low_hz, freq_high_hz]*2/sr);
%freq_hz = 60;
%freq_rel = freq_hz ./ sr;
%[b, a] = ellip(2, 0.5, 20, freq_hz*2/sr, 'high');
data_highpass = filtfilt(b, a, double(data));

sound(data_highpass(1E7+(1:1E6))./35, sr);

% regular manual inspection:
for i=1:length(start_inds)
    [i, start_inds(i)]
    sound(data_highpass(start_inds(i):(start_inds(i)+sr.*0.5))./35, sr, 16);
    pause;
end

suspected_inds = [98
                  114
                  117
                  119
                  166
                  168
                  250
                  307
                  326
                 ];

% close-look inspection:
for i=1:length(suspected_inds)
    [i, suspected_inds(i)]
    cur_data = data_highpass((start_inds(suspected_inds(i))-1*sr): ...
                             (start_inds(suspected_inds(i))+sr))./35;
    plot(((-fix(length(cur_data)./2)):(fix(length(cur_data)./2)))./30000, cur_data);
    drawnow;
    if (suspected_inds(i) > 1)
        delta_t = start_inds(suspected_inds(i)) - ...
                  start_inds(suspected_inds(i)-1);
        fprintf('dt = %.3fs\n', delta_t ./ sr);
    end
    sound(cur_data, sr, 16);
    pause;
end

% looking for the missing beeps:
load new_start_inds.mat;
suspected_inds = find(diff(new_start_inds) > 4*sr);  % >4s may contain a
                                                     % missing trigger.
suspected_inds = suspected_inds(3:end);
accum_manual_inputs = [];
for i=1:(length(suspected_inds)-1)
    [i, suspected_inds(i)]
    % the suspected beep cannot occur 2s after the 1st one, or 2s before
    % the second:
    cur_inds = (new_start_inds(suspected_inds(i))  +1*sr): ...
               (new_start_inds(suspected_inds(i)+1)-2*sr);
    fprintf('Start: %d,   Len: %d  <=>  %.3fs\n', cur_inds(1), ...
            cur_inds(end)-cur_inds(1)+1, (cur_inds(end)-cur_inds(1)+1)./sr);
    cur_data = data_highpass(cur_inds)./35;
    plot(cur_inds, cur_data);
    drawnow;
    if (suspected_inds(i) > 1)
        delta_t = new_start_inds(suspected_inds(i)) - ...
                  new_start_inds(suspected_inds(i)-1);
        fprintf('dt = %.3fs\n', delta_t ./ sr);
    end
    sound(cur_data, sr, 16);
    ask_for_mark = true;
    while (ask_for_mark)
        r = input('Mark time? ', 's');
        if (strcmp(upper(r), 'Y') || strcmp(upper(r), 'Yes'))
            [x, y] = ginput(1);
            accum_manual_inputs = [accum_manual_inputs, x];
            fprintf('%d\n', x);
            ask_for_mark = true;
        else
            ask_for_mark = false;
        end
    end
end

sorted_start_inds = sort([new_start_inds, accum_manual_inputs]);
save sorted_start_inds sorted_start_inds;
