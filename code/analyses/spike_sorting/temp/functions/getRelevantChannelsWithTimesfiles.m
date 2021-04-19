function channels=getRelevantChannelsWithTimesfiles(rootDir)
fileNames=dir(fullfile(rootDir,'times*.mat'));
for i = 1:length(fileNames)
    channels(i)=str2num(fileNames(i).name(regexp(fileNames(i).name,['\d'])));
end
channels=sort(channels);
end
