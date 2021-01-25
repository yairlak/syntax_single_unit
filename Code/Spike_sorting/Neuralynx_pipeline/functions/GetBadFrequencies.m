function [BadFrequencies, th] = GetBadFrequencies(signal, SamplingRate)

% Authors: Roy Mukamel and Ariel Tankus.
% Created: 01.08.2007.


    low_freq = 1000;    % frequency boundries for detection of bad frequencies
    high_freq = 3000;
    NumSigmas = 30; % number of stds from mean for setting threshold
    DistanceBetweenPeaks = 10;  % in sample points
    
    [f, amp] = GetPower(signal, SamplingRate);
    
    index = find(f > low_freq & f < high_freq);
    th=mean(amp(index)) + NumSigmas*std(amp(index));
    gt_th = find(amp(index) > th);

    d = diff(gt_th);

    ind_d = [0, find(d > DistanceBetweenPeaks), length(gt_th)];    % discretization of peaks
    BadFrequencies = [];
    for i=2:length(ind_d)
        BadFrequencies = [BadFrequencies; max(f(index(gt_th((ind_d(i-1)+1):ind_d(i)))))];
    end


    
