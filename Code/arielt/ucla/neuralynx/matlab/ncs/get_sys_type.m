function [sys_type, TTL_fname] = get_sys_type()
% get_sys_type    

% Author: Ariel Tankus.
% Created: 17.02.2010.


%                   {TTL_filename,     Label}
recording_systems = {'Events.Nev',     'Neuralynx';
                     'nlx_new.nev',    'Neuralynx-New';
                     'neuroport.nev',  'Neuroport';
                     'spike2_ttl.mat', 'Spike2';
                     'ao_ttl.mat',     'Alpha-Omega'};

sys_type = 'Unknown';
if (nargout >= 2)
    TTL_fname = '/dev/null';
end

for i=1:size(recording_systems, 1)
    if (exist(['.', filesep, recording_systems{i, 1}], 'file'))
        sys_type = recording_systems{i, 2};
        if (nargout >= 2)
            TTL_fname = recording_systems{i, 1};
        end
    end
end
