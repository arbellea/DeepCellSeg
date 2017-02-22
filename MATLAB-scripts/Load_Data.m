function [data] = Load_Data(data_path,extention,varargin)
% This function will load the data for the tracking algorithm
% Input:
%       - data_path: The location of the img files
%       - extension: The type of file, example: extention = '*.png'
% Output:
%       - data: A sturct including all the data and properites

%% Init

s_in = length(varargin);
ylim = [];
xlim = [];
zlim = [];
%{
switch s_in
    case 0
        
        
    case 2
        if ~isempty(varargin{1})
            ylim = varargin{1}(1):varargin{1}(2);
        end
        if ~isempty(varargin{2})
            xlim = varargin{2}(1):varargin{2}(2);
        end
        zlim = [];
    case 3
        if ~isempty(varargin{1})
            ylim = varargin{1}(1):varargin{1}(2);
            
        end
        if ~isempty(varargin{2})
            xlim = varargin{2}(1):varargin{2}(2);
        end
        if ~isempty(varargin{3})
            zlim = varargin{3}(1):varargin{3}(2);
        end
    otherwise
        error('Problem with Inputs!!!');
        return
        
end

%}
if ~isempty(varargin)
    
expr = varargin{1};
else 
    expr = '\w+(\d+).*';
end

data = struct([]);

%% Read all files
try
    file_list = dir(fullfile(data_path,extention));
catch err
    error('prog:dataPathCheck','Invalid Data Path!!! %s',fullfile(data_path,extention));
end
%disp(['Full File: ' ,fullfile(data_path,extention)]);
%disp(length(file_list));
fileNames = {file_list.name};
tokens = cellfun(@(str) regexp(str,expr,'tokens'),fileNames,'uniformoutput',false);
validFiles = cellfun(@(t)~isempty(t),tokens);
tokens = tokens(validFiles);
fileNames = fileNames(validFiles);
t = cellfun(@(token) str2double(token{1}{1}),tokens);
[~,sortID] = sort(t);
sorted_list = fileNames(sortID);
data(1).Frame_Num = length(sorted_list);
for l = 1:length(sorted_list);
        try
        if strcmp(extention,'*.tif')
            info = imfinfo(fullfile(data_path,sorted_list{l}));
            num_images = numel(info);
                        
            if l==1
                if num_images>1
                for k = 1:num_images
                    I(:,:,k) = imread(fullfile(data_path,sorted_list{l}), k);
                end
                else
                    I = imread(fullfile(data_path,sorted_list{l}));
                end
                    
                
                [H,W,D] = size(I);
                if isempty(ylim)
                    ylim = 1:H;
                end
                if isempty(xlim)
                    xlim = 1:W;
                end
                if isempty(zlim)
                    zlim = 1:D;
                end
            end
            
            data(1).Frame_name{l}=fullfile(data_path,sorted_list{l});
            
            
        else
            
            if l==1
                 if strcmp(extention,'*.txt')
                    I = csvread(fullfile(data_path,sorted_list{l}));
                else
                    I = imread(fullfile(data_path,sorted_list{l}));
                end
                [H,W,D] = size(I);
                if isempty(ylim)
                    ylim = 1:H;
                end
                if isempty(xlim)
                    xlim = 1:W;
                end
                if isempty(zlim)
                    zlim = 1:D;
                end
            end
            data(1).Frame_name{l}=fullfile(data_path,sorted_list{l});
            
        end
    catch err
        fprintf(['Unable to read frame number: ',num2str(l),'\n'])
        data(1).Width = [];
        data(1).Height = [];
        data(1).Depth = [];
        
        return;
    end
    
end
data(1).Type = class(I);
data(1).Width = length(xlim);
data(1).Height = length(ylim);
data(1).Depth = length(zlim);

%% Change path back to previous location

