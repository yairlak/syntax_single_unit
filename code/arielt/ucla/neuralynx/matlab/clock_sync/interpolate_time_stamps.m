function interpolate_time_stamps(cheetah_times_relative, ...
            neuroball_times_relative, indices_cheetah, indices_neuroball)
% interpolate_time_stamps    

% Author: Ariel Tankus.
% Created: 12.04.2005.


if (length(indices_neuroball) ~= length(indices_cheetah))
    error(sprintf(['indices_cheetah and indices_neuroball should have ', ...
                   'identical lengths.\nlength(indices_cheetah) = %d, ', ...
                   'length(indices_neuroball) = %d\n'], ...
                length(indices_cheetah), length(indices_neuroball)));
end

b = regress(cheetah_times_relative(indices_cheetah), ...
            [neuroball_times_relative(indices_neuroball), ...
             ones(length(indices_cheetah), 1)]);

fprintf('Linear coefficients:  a = %.14g, b = %.14g\n', b(1), b(2));

linear_scale_mouse_recording(b);
