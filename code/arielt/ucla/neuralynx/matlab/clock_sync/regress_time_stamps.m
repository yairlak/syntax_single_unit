function [b, b_w_scaling] = regress_time_stamps(cheetah_times_relative, ...
            neuroball_times_relative, indices_cheetah, indices_neuroball)
% regress_time_stamps    

% Author: Ariel Tankus.
% Created: 12.04.2005.


if (length(indices_neuroball) ~= length(indices_cheetah))
    error(sprintf(['indices_cheetah and indices_neuroball should have ', ...
                   'identical lengths.\nlength(indices_cheetah) = %d, ', ...
                   'length(indices_neuroball) = %d\n'], ...
                length(indices_cheetah), length(indices_neuroball)));
end

% regress the deltas instead of the actual numbers, to reduce error due to
% machine accuracy when using absolute times:
% Y-Y0 = [(X-X0), 1]*B
% Y-Y0 = (X-X0)*b1 + 1*b2
% ==> Y = X*b1 - X0*b1 + 1*b2 + Y0
% ==> Y = X*b1 + (Y0-X0*b1) + 1*b2
% ==> Y = [X, 1]*[b1; b2+Y0-X0*b1]
d_ch = cheetah_times_relative(indices_cheetah) - cheetah_times_relative(indices_cheetah(1));
d_pa = neuroball_times_relative(indices_neuroball) - neuroball_times_relative(indices_neuroball(1));
b_w_scaling = regress(d_ch, [d_pa, ones(length(indices_cheetah), 1)]);

% restore b(2) to be related to the absolute regression, rather than the
% relative one:    Y = [X, 1]*[b1; b2+Y0-X0*b1] 
b_w_scaling(2) = b_w_scaling(2) + cheetah_times_relative(indices_cheetah(1)) - ...
    neuroball_times_relative(indices_neuroball(1))*b_w_scaling(1);

b = [1;
     mean(cheetah_times_relative(indices_cheetah) - ...
          neuroball_times_relative(indices_neuroball))];

%b = regress(cheetah_times_relative(indices_cheetah), ...
%            [neuroball_times_relative(indices_neuroball), ...
%             ones(length(indices_cheetah), 1)]);

fprintf('Linear coefficients:  a = %.14g, b = %.14g\n', b_w_scaling(1), b_w_scaling(2));
