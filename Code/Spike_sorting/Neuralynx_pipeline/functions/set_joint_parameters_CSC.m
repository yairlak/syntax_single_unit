function par = set_joint_parameters_CSC(sr)
% set_joint_parameters_CSC    

% Author: Ariel Tankus.
% Created: 25.11.2005.


% ##################################

% SYSTEM
%par.system = 'windows';
par.system = 'linux';

% SPC PARAMETERS
par.mintemp = 0.00;                  % minimum temperature for SPC
par.maxtemp = 0.201;                 % maximum temperature for SPC
par.tempstep = 0.01;                 % temperature steps
par.SWCycles = 100;                  % SPC iterations for each temperature
par.KNearNeighb=11;                  % number of nearest neighbors for SPC
par.num_temp = floor((par.maxtemp-par.mintemp)/par.tempstep);
par.min_clus = 30;                   % minimun size of a cluster
par.max_clus = 13;                    % maximum number of clusters allowed
par.fname = 'data';                  % filename for interaction with SPC
%par.temp_plot = 'lin';               % temperature plot in linear scale
par.temp_plot = 'log';               % temperature plot in log scale

% DETECTION PARAMETERS
par.tmax= 'all';                     % maximum time to load
%par.tmax= 180;                      % maximum time to load
par.tmin= 0;                         % starting time for loading
par.sr=sr;                           % sampling rate
[par.w_pre, par.w_post] = w_pre_post_by_sr(sr);
ref = 3;                           % minimum refractory period (in ms)
par.ref = floor(ref *sr/1000);       % conversion to datapoints
par.stdmin = 5;                      % minimum threshold for detection
par.stdmax = 50;                     % maximum threshold for detection
par.fmin = 300;                      % high pass filter.
par.fmax = 1000;                     % low pass filter.
%par.detection = 'pos';               % type of threshold
par.detection = 'neg';
% par.detection = 'pos';
par.segments = 4;                    % nr. of segments in which the data is cutted.

% INTERPOLATION PARAMETERS
par.int_factor = 2;                  % interpolation factor
par.interpolation = 'y';             % interpolation with cubic splines
%par.interpolation = 'n';


% FEATURES PARAMETERS
par.inputs=10;                       % number of inputs to the clustering
par.scales=4;                        % number of scales for the wavelet decomposition
par.features = 'wav';                % type of feature
%par.features = 'pca';               
if strcmp(par.features,'pca'); par.inputs=3; end


% FORCE MEMBERSHIP PARAMETERS
par.template_sdnum = 3;             % max radius of cluster in std devs.
par.template_k = 10;                % # of nearest neighbors
par.template_k_min = 10;            % min # of nn for vote
%par.template_type = 'mahal';        % nn, center, ml, mahal
par.template_type = 'center';        % nn, center, ml, mahal
par.force_feature = 'spk';          % feature use for forcing (whole spike shape)
%par.force_feature = 'wav';         % feature use for forcing (wavelet coefficients).


% TEMPLATE MATCHING
par.match = 'y';                    % for template matching
%par.match = 'n';                    % for no template matching
par.max_spk = 40000;                % max. # of spikes before starting templ. match.
par.min_freq = 0.5;                  % frequency below which the channel is
                                     % discarded [Hz.]

% ARIEL: 8.2.2006:
%par.max_spikes = 100000;               % max. # of spikes to be plotted
% was:
par.max_spikes = 1000;               % max. # of spikes to be plotted
