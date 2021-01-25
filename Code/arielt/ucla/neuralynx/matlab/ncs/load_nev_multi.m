function TTL_per_bit = load_nev_multi

[sys_type, TTL_fname] = get_sys_type;
switch (sys_type)
 case 'Neuralynx',
    [accumTimeStamps, accumTTL] = neuralynx_read_all_TTLs(TTL_fname);
 case 'Neuralynx-New',
    events = getRawTTLs('nlx_new.nev');
    
    % Yair May 2019: remove false TTLs based on their binary form
    num_bits = 16;
    for i = 2:size(events, 1)
        if  events(i, 2) > 0 && events(i-1, 2) > 0
            current_TTL = dec2bin(events(i, 2), num_bits);
            previous_TTL = dec2bin(events(i-1, 2), num_bits);
            current_is_subset_of_previous_binary_form = true;
            for b = 1:num_bits
                if current_TTL(b) == '1' && previous_TTL(b)=='0'
                    current_is_subset_of_previous_binary_form = false;
                    break
                end
            end
            if ~current_is_subset_of_previous_binary_form
                events(i, 2) = 0;
            end
        end
    end
    
    
    non_zero_events = (events(:, 2) > 0);
    accumTimeStamps = events(non_zero_events, 1);
%    accumTTL        = events(:, 2);
    accumTTL        = ones(length(accumTimeStamps), 1);
    accumTTL(2:2:end) = 0;  % simulate up/down TTLs.
 case 'Neuroport',
    [accumTimeStamps, accumTTL] = neuroport_read_all_TTLs(TTL_fname);
 case 'Spike2',
    load(['.', filesep, 'spike2_ttl.mat']);
    TTL_per_bit = {accumTimeStamps};
    return;
 case 'Alpha-Omega',
    load(['.', filesep, 'ao_ttl.mat']);
    TTL_per_bit = {accumTimeStamps};
    return;
 otherwise,
    error('Unknown TTL sys_type %s.', sys_type);
end

if (isempty(accumTTL) || isempty(accumTimeStamps))
    error('No TTLs found.');
end

TTL_per_bit = {};           % we don't know how many bits are actually used.

if (all(accumTTL == accumTTL(1)))
    % only odd/even were detected:
    fprintf('******************************************************\n');
    fprintf('* ONLY ODD/EVEN TTLS WERE RECORDED!!!                *\n');
    fprintf('* Make sure ttl_filter_odd or ttl_filter_even exist! *\n');
    fprintf('******************************************************\n');
    TTL_per_bit = {accumTimeStamps};
end

% iterate the bits and see if each is informative:
for i=1:16
    cur_bit = bitand(accumTTL, 2^(i-1));     % extract current bit.
    % find changes to this bit:
    change_in_bit = bitxor(cur_bit(2:end), cur_bit(1:(end-1)));
    if (any(change_in_bit))
        % +1: because change_in_bits starts from index 2.
        TTL_per_bit = [TTL_per_bit, {accumTimeStamps(1+find(change_in_bit))}];
    end
end
