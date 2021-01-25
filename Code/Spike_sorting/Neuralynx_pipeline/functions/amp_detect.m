function [spikes,thr,index] = amp_detect(x,par)
% Detect spikes with amplitude thresholding. Uses median estimation.
% Detection is done with filters set by fmin_detect and fmax_detect. Spikes
% are stored for sorting using fmin_sort and fmax_sort. This trick can
% eliminate noise in the detection but keeps the spikes shapes for sorting.

% Modified: 23.07.2009 (Ariel, Roy & Omri).  Order of ellip filter
%                      changed from 2 to 4.

sr=par.sr;
w_pre=par.w_pre;
w_post=par.w_post;
ref=par.ref;
detect = par.detection;
stdmin = par.stdmin;
stdmax = par.stdmax;
fmin_detect = par.detect_fmin;
fmax_detect = par.detect_fmax;
fmin_sort = par.sort_fmin;
fmax_sort = par.sort_fmax;

% HIGH-PASS FILTER OF THE DATA
xf=zeros(length(x),1);
[b,a]=ellip(2,0.1,40,[fmin_detect fmax_detect]*2/sr);
xf_detect=filtfilt(b,a,x);
[b,a]=ellip(2,0.1,40,[fmin_sort fmax_sort]*2/sr);
xf=filtfilt(b,a,x);
lx=length(xf);
clear x;


%%

%noise_std_detect = quickselect_median(abs(xf_detect))/0.6745;
%noise_std_sorted = quickselect_median(abs(xf))/0.6745;
noise_std_detect = median(abs(xf_detect))/0.6745;
noise_std_sorted = median(abs(xf))/0.6745;

thr = stdmin * noise_std_detect;        %thr for detection is based on detect settings.
thrmax = stdmax * noise_std_sorted;     %thrmax for artifact removal is based on sorted settings.

index = [];

%%
% Save filterd signal
if par.j == 1
    save(fullfile(par.data_folder, ['Signal_filtered_', num2str(par.channel)]), 'xf_detect', 'thr', 'thrmax');
    f = figure('visible', 'off', 'color', [1 1 1]);
    % st = 0; % in sec
    ed = 100; % in sec
    yy = xf_detect(1:ed*sr);
    xx = (1:length(yy))/sr;
    plot(xx, yy);
    line([0, ed],[thr, thr], 'color', 'g')
    line([0,ed],[-thr, -thr], 'color', 'g')
    line([0, ed],[thrmax, thrmax], 'color', 'r')
    line([0,ed],[-thrmax, -thrmax], 'color', 'r')
    title(sprintf('Channel %i', par.channel))
    xlim([0, ed])
    xlabel('Time [sec]', 'fontsize', 16)
    ylabel('Signal', 'fontsize', 16)

    savefig(f, fullfile(par.data_folder, ['Signal_filtered_', num2str(par.channel), '.fig']))
    saveas(f, fullfile(par.data_folder, ['Signal_filtered_', num2str(par.channel), '.png']), 'png')
end

%% LOCATE SPIKE TIMES
switch detect
 case 'pos'
  xaux = find(xf_detect(w_pre+2:end-w_post-2) > thr) +w_pre+1;
  if (~enforce_ref_period)
      [index, nspk] = find_spike_max(xf, xaux);
  else
      nspk = 0;
      xaux0 = 0;
      for i=1:length(xaux)
          if xaux(i) >= xaux0 + ref
              [maxi iaux]=max((xf(xaux(i):xaux(i)+floor(ref/2)-1)));    %introduces alignment
              nspk = nspk + 1;
              index(nspk) = iaux + xaux(i) -1;
              xaux0 = index(nspk);
          end
      end
  end
 case 'neg'
  xaux = find(xf_detect(w_pre+2:end-w_post-2) < -thr) +w_pre+1;
  if (~enforce_ref_period)
      [index, nspk] = find_spike_max(-xf, xaux);
  else
      nspk = 0;
      xaux0 = 0;
      for i=1:length(xaux)
          if xaux(i) >= xaux0 + ref
              [maxi iaux]=min((xf(xaux(i):xaux(i)+floor(ref/2)-1)));    %introduces alignment
              nspk = nspk + 1;
              index(nspk) = iaux + xaux(i) -1;
              xaux0 = index(nspk);
          end
      end
  end
  
 case 'both'
  if (~enforce_ref_period)
        xaux = find(xf_detect(w_pre+2:end-w_post-2) > thr) +w_pre+1;
        [index1, nspk1] = find_spike_max(xf, xaux);
        xaux = find(xf_detect(w_pre+2:end-w_post-2) < -thr) +w_pre+1;
        [index2, nspk2] = find_spike_max(-xf, xaux);
        nspk = nspk1 + nspk2;
        index = [index1, index2];
  else
        nspk = 0;
        xaux = find(abs(xf_detect(w_pre+2:end-w_post-2)) > thr) +w_pre+1;
        xaux0 = 0;
        for i=1:length(xaux)
            if xaux(i) >= xaux0 + ref
                [maxi iaux]=max(abs(xf(xaux(i):xaux(i)+floor(ref/2)-1)));    %introduces alignment
                nspk = nspk + 1;
                index(nspk) = iaux + xaux(i) -1;
                xaux0 = index(nspk);
            end
        end
  end
end
index = unique(index);      % will also sort the indices.
nspk = length(index);

% SPIKE STORING (with or without interpolation)
ls=w_pre+w_post;
spikes=zeros(nspk,ls+4); 
xf=[xf'; zeros(50,1)];  % XXX Roee change   was xf=[xf  zeros(1,50)] %%%%
for i=1:nspk                          %Eliminates artifacts
    if max(abs( xf(index(i)-w_pre:index(i)+w_post) )) < thrmax               
        spikes(i,:)=xf(index(i)-w_pre-1:index(i)+w_post+2);
    end
end
aux = find(spikes(:,w_pre)==0);       %erases indexes that were artifacts
spikes(aux,:)=[];
index(aux)=[];
        
switch par.interpolation
    case 'n'
        spikes(:,end-1:end)=[];       %eliminates borders that were introduced for interpolation 
        spikes(:,1:2)=[];
    case 'y'
        %Does interpolation
        handles.par = par; % Yair 2017Dec10
        spikes = int_spikes(spikes,handles);
end
