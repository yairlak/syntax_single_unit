function r = range2(mat)
% range2    The range of values in a matrix.
%
%           r = range2(mat)
%           mat       - matrix.
%           r   - 1x2 - [min, max] - minimal and maximal values of mat.
%
%           See also: min2, max2.

% Author: Ariel Tankus.
% Created: 30.12.2003.

r = [min(mat(:)), max(mat(:))];
