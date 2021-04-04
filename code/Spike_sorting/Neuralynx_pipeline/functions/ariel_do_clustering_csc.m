function ariel_do_clustering_csc(data_folder, channels,sr)
% function Do_clustering_CSC(channels,TimeStamps);
% Does clustering from all channels in the .mat file channels. This batch is
% to be used with Neuralynx data. It runs after spikes are detected with Get_spikes_CSC.
if nargin <1
    load channels
end
if (nargin < 3)
    filename = [data_folder '\CSC' num2str(channels(1)) '.Ncs'];
    if (exist(filename, 'file'))
        sr = str2num(read_field_from_ncs_header(channels(1), '-SamplingFrequency'));
    else
         sr = 40000;
    end
end

t = cputime;
print_date;


par = set_joint_parameters_CSC(sr);
handles.par = par;
%handles.par.stab = 0.8;                     %stability condition for selecting the temperature
handles.par.min_clus_abs = 60;              %minimum cluster size (absolute value)
handles.par.min_clus_rel = 0.005;           %minimum cluster size (relative to the total nr. of spikes)

handles.par.force_auto = 'y';               %automatically force membership if temp>3.

save_vars = {'cluster_class', 'par', 'spikes', 'inspk'};

%figure
%set(gcf,'PaperOrientation','Landscape','PaperPosition',[0.25 0.25 10.5 8]) 

failed_channels = [];

for k=1:length(channels)
    
    try
        
        channel=channels(k)
        tic

            %    % LOAD CSC DATA (for ploting)
            %    filename=sprintf('CSC%d.Ncs',channel);
            %    f=fopen(filename,'r');
            %    fseek(f,16384+8+4+4+4,'bof'); % put pointer to the beginning of data
            %    Samples=fread(f,ceil(sr*60),'512*int16=>int16',8+4+4+4);
            %    x=double(Samples(:));
            %    clear Samples; 
            %    
            %    %Gets the gain and converts the data to micro V.
            %    eval(['scale_factor=textread(''CSC' num2str(channel) '.Ncs'',''%s'',41);']);
            %    x=x*str2num(scale_factor{41})*1e6;
            %    
            %    %Filters and gets threshold
            %    [b,a]=ellip(2,0.1,40,[handles.par.fmin handles.par.fmax]*2/(sr));
            %    xf=filtfilt(b,a,x);
            %    thr = handles.par.stdmin * median(abs(xf))/0.6745;
            %    thrmax = handles.par.stdmax * median(abs(xf))/0.6745;
            %    if handles.par.detection=='neg';
            %        thr = -thr;
            %        thrmax = -thrmax;
            %    end

            % LOAD SC DATA
            load(fullfile(data_folder, ['CSC', num2str(channel), '_spikes']));
            %% - EDITED BY YAIR 2018JAN
%             handles.par.fname = ['data_ch' num2str(channel)];   %filename for interaction with SPC
            handles.par.fname = ['data_CSC' num2str(channel)];   %filename for interaction with SPC
            % --end edit
            %%
            
            nspk = size(spikes,1);
            handles.par.min_clus = max(handles.par.min_clus_abs,handles.par.min_clus_rel*nspk);

            % CALCULATES INPUTS TO THE CLUSTERING ALGORITHM. 
            inspk = wave_features(spikes,handles);              %takes wavelet coefficients.
            
            % GOES FOR TEMPLATE MATCHING IF TOO MANY SPIKES.
            if size(spikes,1)> handles.par.max_spk
                naux = min(handles.par.max_spk,size(spikes,1));
                inspk_aux = inspk(1:naux,:);
            else
                inspk_aux = inspk;
            end
            
            %INTERACTION WITH SPC
            save(handles.par.fname,'inspk_aux','-ascii'); 
            [clu, tree] = run_cluster(handles);
            [temp] = find_temp(tree,handles);

            %DEFINE CLUSTERS 
            class1=find(clu(temp,3:end)==0);
            class2=find(clu(temp,3:end)==1);
            class3=find(clu(temp,3:end)==2);
            class4=find(clu(temp,3:end)==3);
            class5=find(clu(temp,3:end)==4);
            class0=setdiff(1:size(spikes,1), sort([class1 class2 class3 class4 class5]));
            whos class*
            
            % IF TEMPLATE MATCHING WAS DONE, THEN FORCE
            if (size(spikes,1)> handles.par.max_spk | ...
                (handles.par.force_auto == 'y' & temp > 3));
                classes = zeros(size(spikes,1),1);
                if length(class1)>=handles.par.min_clus; classes(class1) = 1; end
                if length(class2)>=handles.par.min_clus; classes(class2) = 2; end
                if length(class3)>=handles.par.min_clus; classes(class3) = 3; end
                if length(class4)>=handles.par.min_clus; classes(class4) = 4; end
                if length(class5)>=handles.par.min_clus; classes(class5) = 5; end
                f_in  = spikes(classes~=0,:);
                f_out = spikes(classes==0,:);
                class_in = classes(find(classes~=0),:);
                class_out = force_membership_wc(f_in, class_in, f_out, handles);
                classes(classes==0) = class_out;
                class0=find(classes==0);        
                class1=find(classes==1);        
                class2=find(classes==2);        
                class3=find(classes==3);        
                class4=find(classes==4);        
                class5=find(classes==5);        
            end    
            
            %    %PLOTS
            %    clf
            %    ylimit = [];
            %    subplot(3,5,11)
            %    switch handles.par.temp_plot
            %        case 'lin'
            %            plot([1 handles.par.num_temp],[handles.par.min_clus handles.par.min_clus],'k:',...
            %                1:handles.par.num_temp,tree(1:handles.par.num_temp,5:size(tree,2)),[temp temp],[1 tree(1,5)],'k:')
            %        case 'log'
            %            semilogy([1 handles.par.num_temp],[handles.par.min_clus handles.par.min_clus],'k:',...
            %                1:handles.par.num_temp,tree(1:handles.par.num_temp,5:size(tree,2)),[temp temp],[1 tree(1,5)],'k:')
            %    end
            %    subplot(3,5,6)
            %    hold on
            cluster=zeros(nspk,2);
            cluster(:,2)= index';
            %    num_clusters = length(find([length(class1) length(class2) length(class3)...
            %            length(class4) length(class5) length(class0)] >= handles.par.min_clus));
            %
            %    if length(class0) > handles.par.min_clus; 
            %        subplot(3,5,6); 
            %            max_spikes=min(length(class0),handles.par.max_spikes);
            %            plot(spikes(class0(1:max_spikes),:)','k'); 
            %            xlim([1 size(spikes,2)]);
            %        subplot(3,5,10); 
            %            hold on
            %            plot(spikes(class0(1:max_spikes),:)','k');  
            %            plot(mean(spikes(class0,:),1),'c','linewidth',2)
            %            xlim([1 size(spikes,2)]); 
            %            title('Cluster 0','Fontweight','bold')
            %        subplot(3,5,15)
            %            xa=diff(index(class0));
            %            [n,c]=hist(xa,0:1:100);
            %            bar(c(1:end-1),n(1:end-1))
            %            xlim([0 100])
            %            xlabel([num2str(sum(n(1:3))) ' in < 3ms'])
            %            title([num2str(length(class0)) ' spikes']);
            %    end
            if length(class1) > handles.par.min_clus; 
                %        subplot(3,5,6); 
                %            max_spikes=min(length(class1),handles.par.max_spikes);
                %            plot(spikes(class1(1:max_spikes),:)','b'); 
                %            xlim([1 size(spikes,2)]);
                %        subplot(3,5,7); 
                %            hold
                %            plot(spikes(class1(1:max_spikes),:)','b'); 
                %            plot(mean(spikes(class1,:),1),'k','linewidth',2)
                %            xlim([1 size(spikes,2)]); 
                %            title('Cluster 1','Fontweight','bold')
                %            ylimit = [ylimit;ylim];
                %        subplot(3,5,12)
                %        xa=diff(index(class1));
                %        [n,c]=hist(xa,0:1:100);
                %        bar(c(1:end-1),n(1:end-1))
                %        xlim([0 100])
                %        set(get(gca,'children'),'facecolor','b','linewidth',0.01)    
                %        xlabel([num2str(sum(n(1:3))) ' in < 3ms'])
                %        title([num2str(length(class1)) ' spikes']);
                cluster(class1(:),1)=1;
            end
            if length(class2) > handles.par.min_clus; 
                %        subplot(3,5,6); 
                %            max_spikes=min(length(class2),handles.par.max_spikes);
                %            plot(spikes(class2(1:max_spikes),:)','r');  
                %            xlim([1 size(spikes,2)]);
                %        subplot(3,5,8); 
                %            hold
                %            plot(spikes(class2(1:max_spikes),:)','r');  
                %            plot(mean(spikes(class2,:),1),'k','linewidth',2)
                %            xlim([1 size(spikes,2)]); 
                %            title('Cluster 2','Fontweight','bold')
                %            ylimit = [ylimit;ylim];
                %        subplot(3,5,13)
                %            xa=diff(index(class2));
                %            [n,c]=hist(xa,0:1:100);
                %            bar(c(1:end-1),n(1:end-1))
                %            xlim([0 100])
                %            set(get(gca,'children'),'facecolor','r','linewidth',0.01)    
                %            xlabel([num2str(sum(n(1:3))) ' in < 3ms'])
                cluster(class2(:),1)=2;
                %            title([num2str(length(class2)) ' spikes']);
            end
            if length(class3) > handles.par.min_clus; 
                %        subplot(3,5,6); 
                %            max_spikes=min(length(class3),handles.par.max_spikes);
                %            plot(spikes(class3(1:max_spikes),:)','g');  
                %            xlim([1 size(spikes,2)]);
                %        subplot(3,5,9); 
                %            hold
                %            plot(spikes(class3(1:max_spikes),:)','g');  
                %            plot(mean(spikes(class3,:),1),'k','linewidth',2)
                %            xlim([1 size(spikes,2)]); 
                %            title('Cluster 3','Fontweight','bold')
                %            ylimit = [ylimit;ylim];
                %        subplot(3,5,14)
                %            xa=diff(index(class3));
                %            [n,c]=hist(xa,0:1:100);
                %            bar(c(1:end-1),n(1:end-1))
                %            xlim([0 100])
                %            set(get(gca,'children'),'facecolor','g','linewidth',0.01)    
                %            xlabel([num2str(sum(n(1:3))) ' in < 3ms'])
                cluster(class3(:),1)=3;
                %            title([num2str(length(class3)) ' spikes']);
            end
            if length(class4) > handles.par.min_clus; 
                %        subplot(3,5,6); 
                %            max_spikes=min(length(class4),handles.par.max_spikes);
                %            plot(spikes(class4(1:max_spikes),:)','c');  
                %            xlim([1 size(spikes,2)]);
                cluster(class4(:),1)=4;
            end
            if length(class5) > handles.par.min_clus;  
                %        subplot(3,5,6); 
                %            max_spikes=min(length(class5),handles.par.max_spikes);
                %            plot(spikes(class5(1:max_spikes),:)','m');  
                %            xlim([1 size(spikes,2)]);
                cluster(class5(:),1)=5;
            end
            %
            %% Rescale spike's axis 
            %    if ~isempty(ylimit)
            %        ymin = min(ylimit(:,1));
            %        ymax = max(ylimit(:,2));
            %    end
            %    if length(class1) > handles.par.min_clus; subplot(3,5,7); ylim([ymin ymax]); end
            %    if length(class2) > handles.par.min_clus; subplot(3,5,8); ylim([ymin ymax]); end
            %    if length(class3) > handles.par.min_clus; subplot(3,5,9); ylim([ymin ymax]); end
            %    if length(class0) > handles.par.min_clus; subplot(3,5,10); ylim([ymin ymax]); end
            %        
            %    subplot(3,1,1)
            %    box off; hold on
            %    plot((1:sr*60)/sr,xf(1:sr*60))
            %    line([0 60],[thr thr],'color','r')
            %    ylim([-thrmax/2 thrmax])
            %    title([pwd '   Channel  ' num2str(channel)],'Fontsize',14)    
        toc

        %SAVE FILES
        par = handles.par;
        cluster_class = cluster;
        outfile=fullfile(data_folder, ['times_CSC' num2str(channel)]);
        if (exist('time0', 'var'))
            save_vars = [save_vars, {'time0'}];
        end
        if (exist('timeend', 'var'))
            save_vars = [save_vars, {'timeend'}];
        end
        save(outfile, save_vars{:});
    
    catch ex
        ex.message
        error(ex.message)
        fprintf('!!!!!!!!!!!!!!!!!!!!\n')
        fprintf('!!!!!!!!!!!!!!!!!!!!\n')
        fprintf('!!!!!!!!!!!!!!!!!!!!\n')
        failed_channels = [failed_channels, channel];
    end
    
end

if (~isempty(failed_channels))
    fprintf('\n\nClustering channel');
    if (length(failed_channels) > 1)
        fprintf('s');
    end
    fprintf(' %d', failed_channels);
    fprintf(' failed!\n\n');
end

print_cpu_time(t, length(channels));
