function cell_wo_prefix = rm_prefix_str_cell(cell_of_str, prefix_str)
% rm_prefix_str_cell    Remove a prefix the strings in a cell array of
%                       strings.
%
%                       cell_wo_prefix = rm_prefix_str_cell(cell_of_str,
%                                                           prefix_str)
%                       cell_of_str - cell of strings - each string begins
%                                          with `prefix_str'.
%                       prefix_str  - string - string to remove from the
%                                          beginning of each string in
%                                          `cell_of_str'.
%                       cell_wo_prefix - cell of strings - strings have no
%                                          prefix.

% Author: Ariel Tankus.
% Created: 16.08.2011.


cell_wo_prefix = cellfun(@(x) x((length(prefix_str)+1):end), cell_of_str, ...
                         'UniformOutput', false);
