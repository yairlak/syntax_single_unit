
function extract_high_gamma(patient, data_type, outfs)
    if nargin == 2
        outfs = 1000;
    end
    addpath('functions/high_gamma')
    path2data = ['../../Data/UCLA/patient_' patient '/Raw/' data_type, '/CSC_mat/'];
    mat_files = dir([path2data 'CSC*.mat']);
    for i = 1:length(mat_files)
        if strcmp(mat_files(i).name, 'CSC0.mat'); continue; end % Skip MICROPHONE
        FullPath2MatFile = fullfile(mat_files(i).folder, mat_files(i).name);
        fprintf('Processing file: %s\n', FullPath2MatFile)
        % Load data
        data_struct = load(FullPath2MatFile);
        infs = 1/data_struct.samplingInterval;
        curr_data = data_struct.data;
        % Get stats
        IQR = prctile(curr_data, 75) - prctile(curr_data, 25);
        median = prctile(curr_data, 50);
        % Robust scaling of data
        data_scaled = (curr_data - median)/IQR;
        % clip data
        curr_data(data_scaled > 5) = median + 3*IQR;
        curr_data(data_scaled < -5) = median - 3*IQR;
        % Extract high-gamma (Hilbert transform)
        data = EcogExtractHighGamma(double(curr_data), infs, outfs);
        sfreq = outfs;
        % save to mat file
        fn = fullfile(mat_files(i).folder, ['HighGamma_' mat_files(i).name]);
        save(fn, 'data', 'sfreq', '-v7')
        fprintf('Mat file with high-gamma signal was saved to: %s\n', fn)
    end
end
