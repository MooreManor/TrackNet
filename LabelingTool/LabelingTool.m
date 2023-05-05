function varargout = LabelingTool(varargin)
    gui_Singleton = 1;
    gui_State = struct('gui_Name',       mfilename, ...
        'gui_Singleton',  gui_Singleton, ...
        'gui_OpeningFcn', @LabelingTool_OpeningFcn, ...
        'gui_OutputFcn',  @LabelingTool_OutputFcn, ...
        'gui_LayoutFcn',  [] , ...
        'gui_Callback',   []);
    if nargin && ischar(varargin{1})
        gui_State.gui_Callback = str2func(varargin{1});
    end

    if nargout
        [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
    else
        gui_mainfcn(gui_State, varargin{:});
    end
% End initialization code - DO NOT EDIT
end

% --- Outputs from this function are returned to the command line.
function varargout = LabelingTool_OutputFcn(hObject, eventdata, handles) 
    varargout{1} = handles.output;
    
% initialization code 
end


% --- Executes just before LabelingTool is made visible.
function LabelingTool_OpeningFcn(hObject, eventdata, handles, varargin)
    handles.output = hObject;
    set(gcf,'pointer','fullcrosshair')
    guidata(hObject, handles);
    
end


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
    global idx imgs Visibility Status CoordinateX CoordinateY draw_circle;

    dName_img = uigetdir();
    if dName_img ~= 0
        set(handles.text2, 'String', dName_img);   % set path on ui

        idx = 1;% current frame index
        CoordinateX(1:10000)= -1; % Array to save each frame Coordinate X
        CoordinateY(1:10000)= -1; % Array to save each frame Coordinate Y
        Status(1:10000)= 0; % Array to save each frame status
        Visibility(1:10000)= 1; % Array to save each frame visibility
        draw_circle = false; % Flag to check if the point be show on axes

        % save all files path string array
        imgs = getAllFiles(dName_img);
        guidata(hObject, handles);

        
        axes(handles.axes1); % set current axes to axes1
        %show image on axes
        imageHandle = imshow(imgs{idx});
        
        % assign the ButtonDownFcn to the image handle
        set(imageHandle,'ButtonDownFcn',@ImageClickCallback);
        
        % show path on ui
        set(handles.text3, 'String', imgs{idx});   

        %check if Label.csv in this path, if yes, load and show the data 
        if exist(strcat(dName_img,'/Label.csv'), 'file') == 2 % value 2 means file exist
            fid = fopen(strcat(dName_img,'/Label.csv'));
            out = textscan(fid,'%s%d%f%f%d','Headerlines',1,'delimiter',',');
            fclose(fid);

            for i = 1:size(imgs)
                Visibility(i) = out{2}(i);
                CoordinateX(i) = out{3}(i);
                CoordinateY(i) = out{4}(i);
                Status(i) = out{5}(i);
            end
        end
    end

end


% Get mouse click coordinates function, tutorial: https://stackoverflow.com/questions/14684577/matlab-how-to-get-mouse-click-coordinates/14685313
function ImageClickCallback ( objectHandle , eventData)

    global idx CoordinateX CoordinateY draw_circle Circle

    axesHandle  = get(objectHandle,'Parent');
    
    % get mouse click coordinates x, y
    coordinates = get(axesHandle,'CurrentPoint');  
    x = coordinates(1,1);
    y = coordinates(1,2);

    % save x, y to global variable Coordinate
    CoordinateX(idx) = round(x);
    CoordinateY(idx) = round(y);
    
    % since only a circle can be plot in axes, so we need to check if we had draw the circle in axes, if yes, we need to delete it
    radius = 1;
    if draw_circle == true
        delete(Circle);
    end
    
    %draw the circle on axes
    Circle = viscircles([x y],radius,'Color','r');
    draw_circle = true;

end



% get all *.jpg file function, refer to https://cn.mathworks.com/matlabcentral/fileexchange/48238-nonstationary-extreme-value-analysis--neva--toolbox?focused=5028815&tab=function
function fileList = getAllFiles(dirName)
    dirData = dir(strcat(dirName,'/*.jpg'));   
    dirIndex = [dirData.isdir]; 
    fileList = {dirData(~dirIndex).name}';  
    
    if ~isempty(fileList)
        fileList = cellfun(@(x) fullfile(dirName,x),fileList,'UniformOutput',false);
    end
    
    subDirs = {dirData(dirIndex).name};  
    validIndex = ~ismember(subDirs,{'.','..'});    
    
    for iDir = find(validIndex)                  
        nextDir = fullfile(dirName,subDirs{iDir});    
        fileList = [fileList; getAllFiles(nextDir)];  
    end
    
end



% --- Executes on key press with focus on figure1
function figure1_WindowKeyPressFcn(hObject, eventdata, handles)

    global idx imgs draw_circle CoordinateX CoordinateY Circle Visibility Status 
    
    switch eventdata.Key
        case 'leftarrow' % show previous frame 
            if idx > 1  % idx range should be 1~lenght(imgs)
                idx = idx - 1; 
                guidata(hObject, handles);
                % set current axes to axes1
                axes(handles.axes1); 
                %show image on axes
                imageHandle = imshow(imgs{idx});
                % assign the ButtonDownFcn to the image handle
                set(imageHandle,'ButtonDownFcn',@ImageClickCallback);

                % default all of radio buttons is enable
                set(handles.frying, 'Enable', 'on');   
                set(handles.hit, 'Enable', 'on');   
                set(handles.bouncing, 'Enable', 'on');   


                if Visibility(idx) == 0 % disable all of radio buttons
                    set(handles.no_ball, 'Value', 1);
                    set(handles.frying, 'Enable', 'off');   
                    set(handles.hit, 'Enable', 'off');   
                    set(handles.bouncing, 'Enable', 'off');   
                elseif Visibility(idx) == 1 % set easy_identification radio button to 1
                    set(handles.easy_identification, 'Value', 1);
                elseif Visibility(idx) == 2 % set hard_identification radio button to 1
                    set(handles.hard_identification, 'Value', 1);
                elseif Visibility(idx) == 3 % set occluded_ball radio button to 1
                    set(handles.occluded_ball, 'Value', 1);
                end

                if Status(idx) == 0 % set frying radio button to 1
                    set(handles.frying, 'Value', 1);
                elseif Status(idx) == 1 % set hit radio button to 1
                    set(handles.hit, 'Value', 1);
                elseif Status(idx) == 2 % set bouncing radio button to 1
                    set(handles.bouncing, 'Value', 1);
                end

                set(handles.text3, 'String', imgs{idx}); % show path on text3

                if CoordinateX(idx)==-1 && CoordinateY(idx)==-1
                    set(handles.text3, 'ForegroundColor', 'r'); % for reminding, if the frame do not be labeled, the path will be red.
                    draw_circle = false;
                else
                    set(handles.text3, 'ForegroundColor', 'b'); % for reminding, if the frame do not be labeled, the path will be red
                    Circle = viscircles([CoordinateX(idx) CoordinateY(idx) ],1,'Color','r');
                    draw_circle = true;
                end
            end

        case 'rightarrow' % show next frame 
            [m,n] = size(imgs); % get images size
            if idx < m % idx range should be 1~lenght(imgs)

                idx = idx + 1;
                guidata(hObject, handles);
                % set current axes to axes1
                axes(handles.axes1);
                %show image on axes
                imageHandle = imshow(imgs{idx});
                % assign the ButtonDownFcn to the image handle
                set(imageHandle,'ButtonDownFcn',@ImageClickCallback);

                % default all of radio buttons is enable
                set(handles.frying, 'Enable', 'on');   
                set(handles.hit, 'Enable', 'on');   
                set(handles.bouncing, 'Enable', 'on');   

                if Visibility(idx) == 0 % disable all of radio buttons
                    set(handles.no_ball, 'Value', 1);
                    set(handles.frying, 'Enable', 'off');   
                    set(handles.hit, 'Enable', 'off');   
                    set(handles.bouncing, 'Enable', 'off');   
                elseif Visibility(idx) == 1 % set easy_identification radio button to 1
                    set(handles.easy_identification, 'Value', 1);
                elseif Visibility(idx) == 2 % set hard_identification radio button to 1
                    set(handles.hard_identification, 'Value', 1);
                elseif Visibility(idx) == 3 % set occluded_ball radio button to 1
                    set(handles.occluded_ball, 'Value', 1);
                end

                if Status(idx) == 0 % set frying radio button to 1
                    set(handles.frying, 'Value', 1);
                elseif Status(idx) == 1 % set hit radio button to 1
                    set(handles.hit, 'Value', 1);
                elseif Status(idx) == 2 % set bouncing radio button to 1
                    set(handles.bouncing, 'Value', 1);
                end


                set(handles.text3, 'String', imgs{idx}); % show path on text3
                if CoordinateX(idx)==-1 && CoordinateY(idx)==-1
                    set(handles.text3, 'ForegroundColor', 'r'); % for reminding, if the frame do not be labeled, the path will be as red.
                    draw_circle = false;
                else
                    set(handles.text3, 'ForegroundColor', 'b'); % for reminding, if the frame do not be labeled, the path will be as blue.
                    Circle = viscircles([CoordinateX(idx) CoordinateY(idx) ],1,'Color','r');
                    draw_circle = true;
                end
            end
    end
    
end


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)

    global imgs CoordinateX CoordinateY Visibility Status 
    
    % for reminding, check if there have Coordinates dont be labeled
    alert = false;
    for i = 1:size(imgs)
        if Visibility(i) ~= 0
            if CoordinateX(i)==-1 || CoordinateY(i) ==-1
                alert = true;
                Path = strsplit(char(imgs(i)),'/');
                fileName = Path{length(Path)};
                warndlg(strcat(fileName, 'Visibility is not No BALL, but Coordinates dont be labeled') ,'Alert');
                break;
            end
        end
    end
    
    % if all of images be labeled correctly, output Label.csv
    if alert == false
        disp(strcat(get(handles.text2, 'String')))
        fid = fopen(strcat(get(handles.text2, 'String'),'\Label.csv'),'w');
        fprintf(fid, 'file name,visibility,x-coordinate,y-coordinate,status\n'); % Label.csv header
        for i = 1:size(imgs)
            Path = strsplit(char(imgs(i)),'\');
            fileName = Path{length(Path)};
            if Visibility(i)== 0
                fprintf(fid, '%s,0,,,\n', fileName);
            else
                fprintf(fid, '%s,%d,%d,%d,%d\n', fileName, Visibility(i), CoordinateX(i), CoordinateY(i), Status(i));

            end
        end
        fclose(fid);
    end


end


% --- Executes when selected object is changed in VisibilityGroup.
function VisibilityGroup_SelectionChangedFcn(hObject, eventdata, handles)
    global idx Visibility 
    
    % default all of radiobutton as enable
    set(handles.frying, 'Enable', 'on');   
    set(handles.hit, 'Enable', 'on');   
    set(handles.bouncing, 'Enable', 'on');   

    switch(get(eventdata.NewValue,'Tag'));
        case 'no_ball' % turn all of radiobutton as disable
            Visibility(idx) = 0;
            set(handles.frying, 'Enable', 'off');   
            set(handles.hit, 'Enable', 'off');   
            set(handles.bouncing, 'Enable', 'off');   
        case 'easy_identification'
            Visibility(idx) = 1;
        case 'hard_identification'
            Visibility(idx) = 2;
        case 'occluded_ball'
            Visibility(idx) = 3;
    end
    
end


% --- Executes when selected object is changed in StatusGroup.
function StatusGroup_SelectionChangedFcn(hObject, eventdata, handles)
    global idx Status 
    switch(get(eventdata.NewValue,'Tag'));
        case 'frying'
            Status(idx)  = 0;
        case 'hit'
            Status(idx)  = 1;
        case 'bouncing'
            Status(idx)  = 2;
    end
end



% --- Executes on key press with focus on figure1 or any of its controls.
function figure1_KeyPressFcn(hObject, eventdata, handles)
end
