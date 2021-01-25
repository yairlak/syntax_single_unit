function [ch, sub_cl, region] = cl_to_ch_sub_cl(cl)
% cl_to_ch_sub_cl    Convert a single-index representation of a cluster into
%                    a representation by two indices: channel no. and number
%                    of cluster within the channel.
%
%                    [ch, sub_cl, region] = cl_to_ch_sub_cl(cl)
%                    cl     - 1xn - a list of cluster numbers.
%                    ch     - 1xn - a list of channel numbers from which the
%                                   clusters were sorted out.
%                    sub_cl - 1xn - a list of indices of the cluster number
%                                   within the same channel number.
%                    region - nx1 - cell - a list of the brain region
%                                   corresponding to the cluster (opt.).
%
%                    See also:  clusters_electrode_montage,
%                               invert_clusters_per_channel_table,
%                               find_brain_region.

% Author: Ariel Tankus.
% Created: 09.11.2006.


c = clusters_electrode_montage;
if (size(clusters_electrode_montage, 2) == 2)
    error(sprintf(['Old format clusters_electrode_montage found.  Run: \n', ...
                   'num_clusters=invert_clusters_per_channel_table(', '''', ...
                   'long_clusters_per_channel.txt', '''', ');\n']));
end

num_cl = length(cl);

ch = zeros(1, num_cl);
sub_cl = zeros(1, num_cl);
if (nargout >= 3)
    region = cell(num_cl, 1);
end

for k=1:num_cl
    for i=1:size(c, 1)

        ind = find(c{i, 1} == cl(k));
        if (~isempty(ind))
            sub_cl(k) = ind;
            ch(k) = c{i, 3};
            if (nargout >= 3)
                region{k} = c{i, 2};
            end
            break;    % jumps to the next outer loop.
        end

    end
end
