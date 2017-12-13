% Convert ISBI to DeepCellSeg format

data_path_base = '/Users/assafarbelle/Documents/ISBI-Data/';
output_path_base = '~'
data_set_name = 'Fluo-N2DH-SIM+' %Should be one of the ISBI data sets
seq = '01' % should be one of the valid sequences: 01 or 02
valid_segmentations = 0:64; %In ISBI training set not all segmentation frames include all the cells, this should include only frames with full segmentations


im_path_base = sprintf('%s/Training/%s/%s/t%03d.tif',data_path_base,data_set_name,seq);
seg_path_base = sprintf('%s/Training/%s/%s_GT/SEG/man_seg%03d.tif',data_path_base,data_set_name,seq);
test_path_base = sprintf('%s/Challenge/%s/%s/t%03d.tif',data_path_base,data_set_name,seq);
net_path_base = sprintg('%s/DataForNet/%s_%s/',output_path_base,data_set_name,seq)


%%
mkdir(net_path_base)
mkdir(fullfile(net_path_base,'Seg'))
mkdir(fullfile(net_path_base,'Raw'))
%%
fidT = fopen(fullfile(net_path_base,'train.csv'),'w');
fidV = fopen(fullfile(net_path_base,'val.csv'),'w');
lenT = floor(numel(valid_segmentations)/2);
clear T2
for i = 1:lenT
    T2(2*i-1) = valid_segmentations(lenT+(i));
    T2(2*i) = valid_segmentations(lenT-(i-1));
    
end 
if mod(numel(valid_segmentations),2)
    T2(end+1) = valid_segmentations(end);
end
m_I = 0
for i = 1:numel(valid_segmentations)
    t = T2(i);
    im_path = sprintf(im_path_base,t);
    seg_path = sprintf(seg_path_base,t);
    S = imread(seg_path);
    I = imread(im_path);
    m_I = max(max(I(:)),m_I)
    continue
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
    if i/numel(valid_segmentations)<0.7
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

    
    
    



