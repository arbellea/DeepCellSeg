%%
%GT_dir = '/home/arbellea/ess/Data/Alon_Full_With_Edge/Val/Seg/';
GT_dir = '/Users/assafarbelle/Documents/PhD/Data/Alon_Full_With_Edge/Val/Seg';
Vis_dir = '/Users/assafarbelle/Documents/PhD/Data/Alon_Full_With_Edge/Val/Vis';
GT_txt = '*.png';
GT_exp = 'Alon_Lab_H1299_t_(\d+)_y_1_x_1.png';
GT_Data = Load_Data(GT_dir,GT_txt,GT_exp);
Vis_Data = Load_Data(Vis_dir,GT_txt,GT_exp);
%%
%Seg_dir = '/home/arbellea/ess/Results/Output/Alon_Full_With_Edge/GAN/num_ex_1_w_edge_fast_switch_lr_0.001/Val/Raw/';

%ex_name = 'ex_11';
%chkpt = 43000;

%ex_name = 'ex_8';
%chkpt = 44500;

%ex_name = 'ex_4';
%chkpt = 44500;
%chkpt = 45000;
%%
%ex_name = 'ex_2_b';
%chkpt = 2500;
%chkpt = 3000;
%chkpt = 3500;
%chkpt = 5500;
%chkpt = 6000;
%chkpt = 6500;
%chkpt = 7000;
%chkpt = 8000;
%chkpt = 14000;
%chkpt = 15500;
%chkpt = 20500; % Rec 52 Perc 60
%chkpt = 22000; % Rec 83 Perc 80 81.5
%chkpt = 24000; % Rec 73 Perc 60
%chkpt = 28000; % Rec 78 Perc 79.7 F 78.8

%chkpt = 24000; % Perc 71 Rec 70. Under Seg
%chkpt = 25500; % Rec 61 Perc 63
%chkpt = 26000; % Rec 42 Perc 51
%chkpt = 26500; % Rec 44 Perc 53
%%
ex_name = 'ex_3_b';
%%chkpt = 5000;
%chkpt = 6000;
%chkpt = 7500;
%chkpt = 8500;
%chkpt = 9500;
%chkpt = 13000;
%chkpt = 14500;
%chkpt = 18500; % Per 71 Rec 71. Little bit under seg
%chkpt = 20000; % Rec 73 Perc 72
%chkpt = 20500; % Rec 79 Perc 78
%chkpt = 22000; % Rec 81 Perc 80 F 80.7
chkpt = 23500; % Goof Rec 83.6 Perc 81.5 F 82.5
%%
%ex_name = 'ex_8_b';
%chkpt = 7500;
%chkpt = 8500;
%chkpt = 9500;
%chkpt = 12500;
%chkpt = 14500;
%chkpt = 15000;
%chkpt = 18000; % Perc 68, Rec 68. Under segmentation
%chkpt = 19500; % Perc 68, Rec 68. Under segmentation
%chkpt = 20500; % Perc 75 Rec 75
%chkpt = 22000; % Perc 73.7 Rec 74.3 F 74
%chkpt = 23500; % Goof Rec 75 Perc 76.6 F 75.8
%%

%ex_name = 'ex_11_b';
%chkpt = 5000;
%chkpt = 6000;
%chkpt = 8000;
%chkpt = 8500;
%chkpt = 12500;
%chkpt = 13500;
%chkpt = 18500; % Not Good At All
%chkpt = 20500; % Goof Rec 86 Perc 85 F 85.5%
%chkpt = 22000; % Goof Rec 85.7 Perc 86.5 F 86.1
%chkpt = 23500; % Goof Rec 86.6 Perc 85.5 F 86.0
%%
ex_name = 'ex_1_c';
%chkpt = 9500;
%chkpt = 10154; % Rec 75 Perc 75.3 F 75.16
%chkpt = 10680; % Rec 74.5 Perc 80.5 F 77.45
%chkpt = 11000; % Rec 76.7 Perc 82.1.3 F 79.37
%chkpt = 11498; % Rec 82.4 Perc 83.9 F 83.1
%chkpt = 11953; % Rec 79.3 Perc 85.3 F 82.2
%chkpt = 12304; % Rec  82.8 Perc 87.5 F 85.14 
%chkpt = 12500; % Rec  82. Perc 86.9 F 84.4 
%chkpt = 12953; % Rec  83.7 Perc 86.8 F 85.2 
%chkpt = 13456; % Rec  83.7 Perc 86.3 F 85.
%chkpt = 13500; % Rec  85.9 Perc 85.8 F 85.4 
%chkpt = 13755; % Rec  85.09 Perc 86.2 F 85.6
%chkpt = 14650; % Rec  87.2 Perc 79.2 F 83.
%chkpt = 16000; % Rec  89. Perc 85.6 F 87.3
%chkpt = 17000; % Rec  88. Perc 82.3 F 85.17
%chkpt = 19600; % Rec  89.9 Perc 82. F 85.7
%chkpt = 20000; % Rec  89.9 Perc 82. F 85.7
%chkpt = 43450; % Rec  89.9 Perc 82. F 85.7
chkpt = 44050; % Rec  90 Perc 58.9 F 71.75

%%
%ex_name = 'ex_2_c';
%chkpt = 2500;
%chkpt = 49500; %Rec 87.2 Perc 85.4 F 86.3
%chkpt = 56000; %Rec 87.2 Perc 85.4 F 86.1
%chkpt = 59725; %Rec 86.8 Perc 83.29 F 84.98
%chkpt = 60050; %Rec 86.8 Perc 83.54 F 85.16
%chkpt = 60150; %Rec 87.2 Perc 84.3 F 85.78
%chkpt = 60450; %Rec 87.2 Perc 83.9 F 85.4
%%
%ex_name = 'ex_4_c';
%chkpt = 18000; %Rec 66.2 Perc 68.9 F 67.5
%%
%ex_name = 'ex_8_c';
%chkpt = 18000; %Rec 78.5 Perc 83.6 F 81
%chkpt = 19000; %Rec 78.5 Perc 82.7 F 80.3
%chkpt = 48000; %Rec 81.5 Perc 84.5 F 83.0

%%
%ex_name = 'ex_4_d';
%chkpt =  5500; %Rec 64.2 Perc 68.5 F 66.2
%chkpt =  6454; %Rec 79.3 Perc 83.8 F 81.5
%chkpt =  6903; %Rec 80.2 Perc 83.5 F 81.8
%chkpt =  7500; %Rec 81.4 Perc 83 F 82.2
%chkpt =  7618; %Rec 83.3 Perc 79.5 F 81.3
%chkpt =  8000; %Rec 80.2 Perc 81.7 F 80.9
%chkpt =  31000; %Rec 86.8 Perc 86.8 F 86.6


%%
%ex_name = 'ex_8_d';
%chkpt = ; %Rec 66.2 Perc 68.9 F 67.5
%chkpt =  26000; %Rec 87.73 Perc 81.3 F 84.39
%%
ex_name = 'num_ex_0.5_real';
%chkpt = ; %Rec 66.2 Perc 68.9 F 67.5
chkpt =  999999; %Rec 87.73 Perc 81.3 F 84.39
%%
ex_name = 'no_advers';
%chkpt = ; %Rec 66.2 Perc 68.9 F 67.5
chkpt =  41500; %Rec 2.43 Perc 67.11 F 4.7
%%
Seg_dir = sprintf('/Users/assafarbelle/GAN_Results/Output/Alon_Full_With_Edge/GAN/%s/model_%d.ckpt/Val/Raw', ex_name,chkpt);
Seg_txt = '*.png';
Seg_exp = 'Alon_Lab_H1299_t_(\d+)_y_1_x_1.png';
Seg_Data = Load_Data(Seg_dir,Seg_txt,Seg_exp);
%%
TP = 0;
tp = zeros(1,Seg_Data.Frame_Num);
j = zeros(1,Seg_Data.Frame_Num);
NGT = 0;
ngt = zeros(1,Seg_Data.Frame_Num);
NS = 0;
ns = zeros(1,Seg_Data.Frame_Num);
%%
c=0;
for t = 1:Seg_Data.Frame_Num
   G = imread(GT_Data.Frame_name{t});
   S = imread(Seg_Data.Frame_name{t});
   I = imread(Vis_Data.Frame_name{t});
   
   Gbw = G==1;
   Sbw = imfill(S(:,:,2)>127*0.5,'holes');
   
   sizeG = size(Gbw);
   sizeS = size(Sbw);
   sizeDiff = (sizeG-sizeS)/2;
   Gbw = Gbw(sizeDiff(1)+1:end-sizeDiff(1),sizeDiff(2)+1:end-sizeDiff(2));
   GL = bwlabel(Gbw,4);
   SL = bwlabel(Sbw,4);
   j(t) = sum(Gbw(:)&Sbw(:))./sum(Gbw(:)|Sbw(:));
   NGT = NGT + max(GL(:));
   NS = NS + max(SL(:));
   
   ngt(t) = max(GL(:));
   ns(t) = max(SL(:));
   for g = 1:max(GL(:))
       Gl = GL==g;
       for s = unique(SL(Gl))'
           if s==0
               continue
           end
           Sl = SL==s;
           if sum(Sl(:))<50
               ns(t) = ns(t)-1;
               NS = NS-1;
               Sbw(Sl) = 0;
               c = c+1
               continue
           end
           intersection = sum(Gl(:)&Sl(:));
           if intersection ==0
               continue;
           end
           union = sum(Gl(:)|Sl(:));
           J = intersection./union;
           if J >= 0.5
               TP = TP+1;
               tp(t) = tp(t)+1;
           else
               J;
           end 
       end     
   end
   figure(t);
subplot(2,3,1);
imshow(I);
a(1) = gca;
subplot(2,3,2);
imshow(Gbw);
a(2) = gca;
subplot(2,3,3);

imshow(Sbw)
a(3) = gca;
subplot(2,3,4);
imshow(S)
a(4) = gca;
subplot(2,3,5);
imshow(imfuse(Gbw,Sbw))
a(5) = gca;
linkaxes(a);
end
%%
j
rec = tp./ngt
prec = tp./ns
Rec = TP./NGT
Prec = TP./NS
F = 2*Prec*Rec./(Prec+Rec)
%%
return;
figure(t);
subplot(3,2,1);
imshow(I);
subplot(3,2,2);
imshow(Gbw);
subplot(3,2,3);
imshow(Sbw)
subplot(3,2,4);
imshow(S)
subplot(3,2,5);
imshow(imfuse(Gbw,Sbw))

