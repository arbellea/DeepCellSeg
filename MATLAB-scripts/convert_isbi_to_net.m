% Convert ISBI to Net
% im_path_base = '/Users/assafarbelle/Documents/ISBI-Data/Training/DIC-C2DH-HeLa/01/t%03d.tif';
% test_path_base = '/Users/assafarbelle/Documents/ISBI-Data/Challenge/DIC-C2DH-HeLa/01/t%03d.tif';
% seg_path_base = '/Users/assafarbelle/Documents/ISBI-Data/Training/DIC-C2DH-HeLa/01_GT/SEG/man_Seg%03d.tif';
% net_path_base = '/Users/assafarbelle/Documents/ISBI-Data/Training/DIC-C2DH-HeLa/01_GT/NET/';
% T = [2,5,21,31,33,39,54];
% im_path_base = '/Users/assafarbelle/Documents/ISBI-Data/Training/DIC-C2DH-HeLa/02/t%03d.tif';
% test_path_base = '/Users/assafarbelle/Documents/ISBI-Data/Challenge/DIC-C2DH-HeLa/02/t%03d.tif';
% seg_path_base = '/Users/assafarbelle/Documents/ISBI-Data/Training/DIC-C2DH-HeLa/02_GT/SEG/man_Seg%03d.tif';
% net_path_base = '/Users/assafarbelle/Documents/ISBI-Data/Training/DIC-C2DH-HeLa/02_GT/NET/';
% T = [6,7,14,27,34,38,42,61,67];


im_path_base = '/home/arbellea/ISBI-Challenge-Data/Training/Fluo-N2DH-SIM+/01/t%03d.tif';
test_path_base = '/home/arbellea/ISBI-Challenge-Data/Challenge/Fluo-N2DH-SIM+/01/t%03d.tif';
seg_path_base = '/home/arbellea/ISBI-Challenge-Data/Training/Fluo-N2DH-SIM+/01_GT/SEG/man_seg%03d.tif';
net_path_base = '/home/arbellea/ISBI-Challenge-Data/Training/Fluo-N2DH-SIM+/01_GT/NET/';
T = 0:64;

% 
% im_path_base = '/home/arbellea/ISBI-Challenge-Data/Training/Fluo-N2DH-SIM+/02/t%03d.tif';
% test_path_base = '/home/arbellea/ISBI-Challenge-Data/Challenge/Fluo-N2DH-SIM+/02/t%03d.tif';
% seg_path_base = '/home/arbellea/ISBI-Challenge-Data/Training/Fluo-N2DH-SIM+/02_GT/SEG/man_seg%03d.tif';
% net_path_base = '/home/arbellea/ISBI-Challenge-Data/Training/Fluo-N2DH-SIM+/02_GT/NET/';
% T = 0:149;

% % 
% im_path_base = '/home/arbellea/ISBI-Challenge-Data/Training/Fluo-C2DL-MSC/01/t%03d.tif';
% test_path_base = '/home/arbellea/ISBI-Challenge-Data/Challenge/Fluo-C2DL-MSC/01/t%03d.tif';
% seg_path_base = '/home/arbellea/ISBI-Challenge-Data/Training/Fluo-C2DL-MSC/01_GT/SEG/man_seg%03d.tif';
% net_path_base = '/home/arbellea/ISBI-Challenge-Data/Training/Fluo-C2DL-MSC/01_GT/NET/';
% T = [5,7,11,19,24,25,29,33,34,35,45];

% im_path_base = '/home/arbellea/ISBI-Challenge-Data/Training/Fluo-C2DL-MSC/02/t%03d.tif';
% test_path_base = '/home/arbellea/ISBI-Challenge-Data/Challenge/Fluo-C2DL-MSC/02/t%03d.tif';
% seg_path_base = '/home/arbellea/ISBI-Challenge-Data/Training/Fluo-C2DL-MSC/02_GT/SEG/man_seg%03d.tif';
% net_path_base = '/home/arbellea/ISBI-Challenge-Data/Training/Fluo-C2DL-MSC/02_GT/NET/';
% T = [0,3,5,6,7,8,9,10,11,12,];


%%
mkdir(net_path_base)
mkdir(fullfile(net_path_base,'Seg'))
mkdir(fullfile(net_path_base,'Raw'))
%%
fidT = fopen(fullfile(net_path_base,'train.csv'),'w');
fidV = fopen(fullfile(net_path_base,'val.csv'),'w');
lenT = floor(numel(T)/2);
clear T2
for i = 1:lenT
    T2(2*i-1) = T(lenT+(i));
    T2(2*i) = T(lenT-(i-1));
    
end 
if mod(numel(T),2)
    T2(end+1) = T(end);
end
for i = 1:numel(T)
    t = T2(i);
    im_path = sprintf(im_path_base,t);
    seg_path = sprintf(seg_path_base,t);
    S = imread(seg_path);
    I = imread(im_path);
    s = size(S);
    RGB = zeros(s(1),s(2),3);
    RGB(:,:,1) = S==0;
    E = zeros(s);
    F = zeros(s);
    for l = unique(S(S>0))'
        BW = imerode(S==l,ones(3));
        E(S==l&(~BW)) = 1;
        F(BW) = 1;
    end
    RGB(:,:,2) = F;
    RGB(:,:,3) = E;
    L = RGB(:,:,2) + RGB(:,:,3)*2;
    out_im_path = sprintf('./Raw/t%03d.png',t);
    out_seg_path = sprintf('./Seg/t%03d.png',t);
    if i/numel(T)<0.7
        fprintf(fidT,'%s,%s\n', out_im_path, out_seg_path);
    else
        
        fprintf(fidV,'%s,%s\n', out_im_path, out_seg_path);
    end
    I = uint16(I);
    imwrite(I, fullfile(net_path_base,out_im_path))
    imwrite(uint8(L),fullfile(net_path_base,out_seg_path))
end
fclose(fidT);

fclose(fidV);
%%
t=0;
fid = fopen(fullfile(net_path_base,'test.csv'),'w');
mkdir(fullfile(net_path_base,'ALL'))
while true
    im_path = sprintf(test_path_base,t);
    if ~exist(im_path,'file')
        fprintf('%s Does not exist.\n Done\n', im_path);
        break
    end
    out_im_path = sprintf('./ALL/t%03d.png',t);
    I = uint16(imread(im_path));
    imwrite(I, fullfile(net_path_base,out_im_path));
    fprintf(fid,'%s,%s\n', out_im_path, out_im_path);
    t = t+1;
end
fclose(fid);

    
    
    



