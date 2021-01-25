function [max_offset, d_cheetah, d_neuroball, cheetah_times_relative, ...
          neuroball_times_relative, d_neuroball_max_offset_mask] = ...
    find_cheetah_neuroball_offset(cheetah_times_relative, ...
            neuroball_times_relative, delta_thresh)
% find_cheetah_neuroball_offset    Find the time offset between a Cheetah
%                                  recording and a mouse recording.
%                                  The recordings are expected to reside in
%                                  the files:
%                                      cheetah_times_relative.txt
%                                      neuroball_times_relative.txt
%
%                                  max_offset = ...
%                                   find_cheetah_neuroball_offset(delta_thresh)
%                                  delta_thresh - 1x1 - a difference between
%                                               Cheetah and Neuroball times
%                                               below this threshold is
%                                               ignored. [microseconds]
%                                  max_offset - 1x1 - the most probable index
%                                               offset between the recordings.
%                                               The offset is to the index of
%                                               the array of Cheetah w.r.t
%                                               Neuroball.
%
%                                  See also: interpolate_time_stamps.

% Author: Ariel Tankus.
% Created: 13.04.2005.


if (nargin < 1)
    delta_thresh = 1500;        % microseconds.
end 
chunk_size = 13;
filter_delta_thresh = 5000;     % microseconds.  If TTLs were
                                % sent/arrived within a time interval
                                % shorter than this, they are removed.

                                
% Yair May 2019
filter_delta_thresh = delta_thresh;

d_cheetah   = diff(cheetah_times_relative);
d_neuroball = diff(neuroball_times_relative);

% Check that there are no sync pulses shorter than delta_thresh
% microsec. apart in Cheetah's recording:
short_d_inds = find(d_cheetah < filter_delta_thresh);
if (~isempty(short_d_inds))
    fprintf(['Found %d sync pulses within %d microsec. of their ' ...
             'predecessors.\nRemoving them.\n'], length(short_d_inds), ...
            delta_thresh);
    cheetah_times_relative(short_d_inds+1) = [];
    d_cheetah = diff(cheetah_times_relative);
    
    fprintf(['Consider the option to create a manual_cheetah_events.txt ' ...
             'file.\n']);
end

% Check that there are no sync pulses shorter than delta_thresh
% microsec. apart in the paradigm's recording:
short_d_inds = find(d_neuroball < filter_delta_thresh);
if (~isempty(short_d_inds))
    fprintf(['Found %d sync pulses within %d microsec. of their ' ...
             'predecessors.\nRemoving them.\n'], length(short_d_inds), ...
            delta_thresh);
    neuroball_times_relative(short_d_inds+1) = [];
    d_neuroball = diff(neuroball_times_relative);
    
    fprintf('Consider the option to manually remove them in the log file.\n');
end

if (chunk_size < length(d_neuroball))
    num_chunks = length(d_neuroball) - chunk_size + 1;
else
    chunk_size = length(d_neuroball);
    num_chunks = 1;
end

max_num_small_delta = -Inf.*ones(num_chunks, 1);
offset = zeros(num_chunks, 1);

if (length(d_cheetah) < chunk_size)
    error(sprintf('Cheetah recording (%d) shorter than chunk size (%d)', ...
                length(d_cheetah), chunk_size));
end

% For each chunk, find the offset which maximizes the # of elements in
% the chunk with small enough offset (i.e., offset < delta_thresh):
for i=1:num_chunks
    
    cur_chunk = d_neuroball(i:(i+chunk_size-1));
    
    for j=1:(length(d_cheetah) - length(cur_chunk) + 1)

        delta_d =  (d_cheetah(j:(j+length(cur_chunk)-1)) - cur_chunk);
        num_small_delta = sum(abs(delta_d) < delta_thresh);

        % for each chunk, keep the offset which maximizes num_small_delta:
        if (num_small_delta > max_num_small_delta(i))
            max_num_small_delta(i) = num_small_delta;
            offset(i) = j - i;
        end
    end
    
end

sorted_table = sortrows([offset, max_num_small_delta]);
max_max_num_small_delta = -Inf;
maj_offset = -Inf;

cur_offset      = sorted_table(1, 1);
max_offset      = cur_offset;
cur_offset_hits = sorted_table(1, 2);
max_offset_hits = cur_offset_hits;

for i=2:num_chunks
    
    if (cur_offset == sorted_table(i, 1))

        % accumulate the number of correct hits for all consecutive
        % chunks of the same offset.
        cur_offset_hits = cur_offset_hits + sorted_table(i, 2);
    
    else

        if (cur_offset_hits > max_offset_hits)
            % update the maximum
            max_offset_hits = cur_offset_hits;
            max_offset = cur_offset;
        end
        
        % start a new accumulation
        cur_offset_hits = 0;
        cur_offset = sorted_table(i, 1);
        
    end
    
end

if (cur_offset_hits > max_offset_hits)
    % update the maximum
    max_offset_hits = cur_offset_hits;
    max_offset = cur_offset;
end

% mark which chunks lead to the max offset.  This will prevent insertion
% of other elements into the regression later on:
d_neuroball_max_offset_mask = false(length(d_neuroball), 1);
max_offset_inds = find(offset == max_offset);
for j=1:length(max_offset_inds)
    i = max_offset_inds(j);
    d_neuroball_max_offset_mask(i:(i+chunk_size-1)) = true;
end
