function find_triggers_in_wav_files(params) 
%% Load data
sentences_file_name = 'sentences_Eng_rand_En02.txt';

%% Draw main figure
f = figure('units', 'normalized', 'outerposition',[0,0,1,1], 'color', [1 1 1]);
myhandles = guihandles(f);
myhandles.chosen_t = [];
myhandles.h_lines = [];
myhandles.onset_times = [];

%% Open text file with all sentences, in the same order as the wav files
myhandles.sentences = importdata(sentences_file_name);

%% Draw big plot
wav_file_name = sprintf('%i.wav', params.wav_file_number); % First stimulus file
[myhandles.y, params.sr] = audioread(fullfile(params.path2stimuli, wav_file_name));
subplot(2,1,1)
myhandles.small_plot=plot((1:length(myhandles.y))/params.sr, myhandles.y);
xlabel('Time [sec]')
myhandles.small_plot_axes = gca;
axis tight
myhandles.params = params;

%% Draw small plot spec
myhandles.small_plot_spec = subplot(2,1,2);
spectrogram(myhandles.y, 128, 120, 1024, myhandles.params.sr,'yaxis');
colorbar('off')

xlabel('Time');ylabel('Frequency')
set(gca, 'xticklabels', [], 'yticklabels', [])
myhandles.small_plot_spec_axes = gca;
axis tight

%% UICONTROLS
myhandles.f = f;
myhandles.static_txt = uicontrol(f,'Style','text',...
                'String',myhandles.sentences(myhandles.params.wav_file_number),...
                'Max',1,'Min',0,...
                'Position',[400 350 800 40], 'fontsize', 14', 'fontweight', 'bold');

% myhandles.static_txt_chosen_t = uicontrol(f,'Style','text',...
%                 'String',myhandles.chosen_t,...
%                 'Max',1,'Min',0,...
%                 'Position',[30 450 150 300], 'fontsize', 14', 'fontweight', 'bold');
           
myhandles.pb_reject = uicontrol(f,'Style','pushbutton','String','Reject',...
                'Position',[660 5 60 20], 'Callback', @pb_reject_Callback);
            
myhandles.pb_sound = uicontrol(f,'Style','pushbutton','String','Play',...
                'Position',[550 5 60 20], 'Callback', @pb_sound_Callback);

myhandles.pb_previous = uicontrol(f,'Style','pushbutton','String','Previous',...
                'Position',[270 5 60 20], 'Callback', @pb_previous_Callback);

myhandles.pb_next = uicontrol(f,'Style','pushbutton','String','Next',...
                'Position',[480 5 60 20], 'Callback', @pb_next_Callback);
            
set(myhandles.small_plot,'ButtonDownFcn',@SmallPlotClickCallback)
set(myhandles.small_plot_spec,'ButtonDownFcn',@SmallPlotSpecClickCallback)

guidata(f,myhandles)

set(f, 'ToolBar', 'figure');
% addlistener(myhandles.small_plot,'ActionEvent',@(hObject, event) draw_vertical_line(hObject,event,t,y,params,myhandles.small_plot));

end

function SmallPlotClickCallback(hObject ,eventData)
   myhandles = guidata(gcbo);
   axesHandle  = get(hObject,'Parent');
   myhandles.coordinates = get(axesHandle,'CurrentPoint'); 
   chosen_t = myhandles.coordinates(1,1);
%    set(myhandles.txtbox, 'string', num2str(chosen_t))
%    
   set(myhandles.f, 'currentaxes', myhandles.small_plot_axes)
   
   ylim = get(axesHandle, 'ylim');
   hold on
   h_line = line([chosen_t chosen_t], [-1 1], 'color', 'r');
   myhandles.h_lines = [myhandles.h_lines, h_line];
   set(gca, 'ylim', ylim)
   drawnow;
   hold off
   myhandles.chosen_t = [myhandles.chosen_t, chosen_t];
%    set(myhandles.static_txt_chosen_t, 'string', myhandles.chosen_t')
   
   guidata(gcbo,myhandles)
end

function pb_reject_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
myhandles = guidata(gcbo);
myhandles.chosen_t = [];
% set(myhandles.static_txt_chosen_t, 'string', myhandles.chosen_t)
guidata(gcbo,myhandles)

for l = 1:length(myhandles.h_lines)
    delete(myhandles.h_lines(l))
end

end

function pb_sound_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
myhandles = guidata(gcbo);

st = 1;
if ~isempty(myhandles.chosen_t)
    st = max(myhandles.params.sr * myhandles.chosen_t);
end
sound(myhandles.y(st:end),myhandles.params.sr)

end

function pb_previous_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

myhandles = guidata(gcbo);
myhandles.onset_times{myhandles.params.wav_file_number} = [];
myhandles.params.wav_file_number = myhandles.params.wav_file_number - 1;

fid = fopen(myhandles.params.output_fname, 'w');
for trial = 1:length(myhandles.onset_times)
    curr_line = [];
    for t = 1:length(myhandles.onset_times{trial})
        curr_line = [curr_line, sprintf('\t%10.3f', myhandles.onset_times{trial}(t))];
    end
    fprintf(fid,'%i\t%s\n', trial, curr_line);
end
fclose(fid);

% Draw big plot
wav_file_name = sprintf('%i.wav', myhandles.params.wav_file_number); % First stimulus file
[myhandles.y, params.sr] = audioread(fullfile(myhandles.params.path2stimuli, wav_file_name));
subplot(2,1,1)
myhandles.small_plot=plot((1:length(myhandles.y))/myhandles.params.sr, myhandles.y);
xlabel('Time [sec]'),ylabel('Stimulus')
myhandles.small_plot_axes = gca;
axis tight
set(myhandles.small_plot,'ButtonDownFcn',@SmallPlotClickCallback)

% Draw small plot spec
myhandles.small_plot_spec = subplot(2,1,2);
spectrogram(myhandles.y, 128, 120, 1024, params.sr,'yaxis');
colorbar('off')
xlabel('Time');ylabel('Frequency')
set(gca, 'xticklabels', [], 'yticklabels', [])
myhandles.small_plot_spec_axes = gca;
axis tight
set(myhandles.small_plot_spec_axes,'ButtonDownFcn',@SmallPlotClickCallback)

myhandles.chosen_t = [];
set(myhandles.static_txt, 'String',myhandles.sentences(myhandles.params.wav_file_number))               
% set(myhandles.static_txt_chosen_t, 'String', myhandles.chosen_t)

guidata(gcbo,myhandles)
end

function pb_next_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
myhandles = guidata(gcbo);
% Add to cell array the last onset times
myhandles.onset_times{myhandles.params.wav_file_number} = myhandles.chosen_t;

% Save current cell of onset times to file
fid = fopen(myhandles.params.output_fname, 'w');
for trial = 1:length(myhandles.onset_times)
    curr_line = [];
    for t = 1:length(myhandles.onset_times{trial})
        curr_line = [curr_line, sprintf('\t%3.6f', myhandles.onset_times{trial}(t))];
    end
    fprintf(fid,'%i\t%s\n', trial, curr_line);
end
fclose(fid);

% Save fig
file_name = sprintf('%i %s', myhandles.params.wav_file_number, myhandles.sentences{myhandles.params.wav_file_number});
IX = strfind(file_name, '?');
if ~isempty(IX)
    file_name(IX:IX)=[];
end
saveas(myhandles.f, fullfile('Figures', file_name), 'png')

% Roll sentence counter by 1
myhandles.params.wav_file_number = myhandles.params.wav_file_number + 1;

% Draw big plot
wav_file_name = sprintf('%i.wav', myhandles.params.wav_file_number); % First stimulus file
[myhandles.y, params.sr] = audioread(fullfile(myhandles.params.path2stimuli, wav_file_name));
subplot(2,1,1)
myhandles.small_plot=plot((1:length(myhandles.y))/myhandles.params.sr, myhandles.y);
xlabel('Time [sec]')
myhandles.small_plot_axes = gca;
axis tight
set(myhandles.small_plot,'ButtonDownFcn',@SmallPlotClickCallback)

% Draw small plot spec
myhandles.small_plot_spec = subplot(2,1,2);
spectrogram(myhandles.y, 128, 120, 1024, params.sr,'yaxis');
colorbar('off')
xlabel('Time');ylabel('Frequency')
set(gca, 'xticklabels', [], 'yticklabels', [])
myhandles.small_plot_spec_axes = gca;
axis tight
set(myhandles.small_plot_spec_axes,'ButtonDownFcn',@SmallPlotClickCallback)


myhandles.chosen_t = [];
set(myhandles.static_txt, 'String',myhandles.sentences(myhandles.params.wav_file_number))               
% set(myhandles.static_txt_chosen_t, 'String', myhandles.chosen_t)
            
guidata(gcbo,myhandles)

end

% 
% 
% function PlotClickCallback(hObject ,eventData)
%    myhandles = guidata(gcbo);
%    axesHandle  = get(hObject,'Parent');
%    myhandles.coordinates = get(axesHandle,'CurrentPoint'); 
%    myhandles.params.curr_t = myhandles.coordinates(1,1);
%    myhandles.params.indexOfInterest = (myhandles.t < myhandles.params.curr_t + myhandles.params.window_size) & (myhandles.t > myhandles.params.curr_t);
%    set(myhandles.f, 'currentaxes', myhandles.small_plot_axes)
%    myhandles.small_plot = plot(myhandles.t(myhandles.params.indexOfInterest),myhandles.y(myhandles.params.indexOfInterest));
%    set(myhandles.small_plot,'ButtonDownFcn',@SmallPlotClickCallback)
%    set(myhandles.small_plot_spec_axes,'ButtonDownFcn',@SmallPlotClickCallback)
%    axis tight
%    
%    set(myhandles.f, 'currentaxes', myhandles.small_plot_spec);
%    spectrogram(myhandles.y(myhandles.params.indexOfInterest), 128, 120, 256, myhandles.params.sr,'yaxis');
%    ylim([0 2500])
% 
%    drawnow;
%    set(myhandles.sld,'Value', myhandles.params.curr_t)
%    guidata(gcbo,myhandles)
% end
% 
% 
% function pb_Callback(hObject, eventdata, handles)
% % hObject    handle to pushbutton1 (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% myhandles = guidata(gcbo);
% curr_txt = get(myhandles.txtbox, 'string');
% fprintf(myhandles.fid, '%s, %s, ', myhandles.static_text, curr_txt);
% set(myhandles.txtbox, 'string', ['Saved(' myhandles.static_text ')']);
% pb_next_Callback
% end
% 
