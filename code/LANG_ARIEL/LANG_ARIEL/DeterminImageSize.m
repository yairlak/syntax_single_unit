function imageRect = DeterminImageSize(Image,winRect)

[M,N,~] = size(Image);

if (N > M)
    imsizepxl1 = winRect(3)/4;
    imsizepxl2 =  imsizepxl1*M/N;
else
    imsizepxl2 = winRect(4)/4;
    imsizepxl1 =  imsizepxl2*N/M;
end
imageRect = [0 0 imsizepxl1 imsizepxl2]; % we want all images to show up imsizepxl x imsizepxl pixels

end