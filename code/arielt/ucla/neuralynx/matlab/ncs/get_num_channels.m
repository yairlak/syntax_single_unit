function n = get_num_channels
% get_num_channels    

% Author: Ariel Tankus.
% Created: 02.08.2009.
% Modified: 28.01.2010.


if (exist(['.', filesep, 'electrode_info.mat'], 'file'))
    % Neuroport:
    load electrode_info.mat;
    n = nchan;
else
    % Neuralynx:

    d = dir('CSC*.Ncs');
    n = length(d);
    if (n > 0)
        return;
    end

    % In online sessions, we may use only the spikes files, w/o .Ncs files:
    d = dir('CSC*_spikes.mat');
    n = length(d);
    
end
