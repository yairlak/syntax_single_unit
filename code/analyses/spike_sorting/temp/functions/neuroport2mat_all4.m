function neuroport_to_mat_all4(infile,outFolder) 
% this functions generates CSC files, all at once 
NS5=openNSx(infile,'precision','double','skipfactor',1e5); % just to extract header data
% save read me file in output folder detailing the ns5 path this file came
% from:
% XXX change so that readme file writes into outfolder directory
mkdir(outFolder);
fid = fopen(fullfile(outFolder,'readme.txt'),'w+');
fprintf(fid,'CSC files were created from this NS5 file:\n')
fprintf(fid,'%s',infile);
fclose(fid);
nchan = double(NS5.MetaTags.ChannelCount);

samp_freq_hz = NS5.MetaTags.SamplingFreq;
samplingInterval =  1000 / samp_freq_hz;


fprintf('Data is split into %d chunks (the pause thing)\n',length(NS5.MetaTags.DataPoints))
% sometimes when more than 128 channels are used a snyc pulse is done
% between the two amps that splits the data into two. Usually you want the
% second chunk in this case. If there is only one chunk, chances are less
% than 128 channels were used (e.g. one amp) and no sync pulse was needed. 
chunkLens=NS5.MetaTags.DataPoints;
for z = 1:length(chunkLens)
    fprintf('Chunk %d is %f secs long\n' , z, chunkLens(z)/samp_freq_hz)
end
% shouldContinue=input('To extract chunk 1 press 1, to extract chunk 2 press 2, to abort press 0\n');
shouldContinue = 1;
switch shouldContinue
    case 0
        return
    case 1
        chunkToExtract = 1;
    case 2
        chunkToExtract = 2;
    case 3
        chunkToExtract = 3;
end

num_samples = double(NS5.MetaTags.DataPoints(chunkToExtract)); % since the second is what I am intersted in




for i = 78:nchan
    data = zeros(1,num_samples); % preallocate for speed
    idx = i% UCLA - NS5.ElectrodesInfo(i).ElectrodeID;
    NS5_1chan=openNSx(infile,['c:' num2str(i)],'precision','int16');
    data=NS5_1chan.Data;
    save(fullfile(outFolder,['CSC' num2str(idx) '.mat']),'data', 'samplingInterval')
    fprintf('CSC of channnel %d saved\n',idx);
    clear NS5_1chan;
end

%% save electrode info 
bytes_per_samp = 2;
chunk_size=NS5.MetaTags.DataPoints(chunkToExtract);
num_chunks=1;
enum = double([NS5.ElectrodesInfo(:).ElectrodeID]);
% is the channel analog input?
is_ainp = strncmp({NS5.ElectrodesInfo(:).Label}, 'ainp', 4);
elec_list = enum(~is_ainp);
num_elec = length(elec_list);
nchan = double(NS5.MetaTags.ChannelCount);
samp_freq_hz = NS5.MetaTags.SamplingFreq;
samplingInterval =  1000 / samp_freq_hz;
period = (1./samp_freq_hz)./(1./30000); % No. of 1/30,000 secs between data
                                        % points e.g. sampling rate of 30
                                        % kS/s = 1; 10 kS/s = 3
time0 = 0;
timeend = round((num_samples-1) .* (1./samp_freq_hz) .* 1E6); %microsec.


% save general info about the electrodes and recording times:
save(fullfile(outFolder,'electrode_info.mat'), 'enum', 'nchan', 'period', 'infile', ...
     'samp_freq_hz', 'bytes_per_samp', 'num_samples', 'chunk_size', ...
     'num_chunks', 'time0', 'timeend', 'is_ainp', 'elec_list', 'num_elec');


end



