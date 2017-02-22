%%
GT_dir = '/home/arbellea/ess/Data/Alon_Full_With_Edge/Val/Seg/';
GT_txt = '*.png';
GT_exp = 'Alon_Lab_H1299_t_(\d+)_y_1_x_1.png';
GT_Data = Load_Data(GT_dir,GT_txt,GT_exp);
%%
Seg_dir = '/home/arbellea/ess/Results/Output/Alon_Full_With_Edge/GAN/num_ex_1_w_edge_fast_switch_lr_0.001/Val/Raw/';
Seg_txt = '*.png';
Seg_exp = 'Alon_Lab_H1299_t_(\d+)_y_1_x_1.png';
Seg_Data = Load_Data(Seg_dir,Seg_txt,Seg_exp);
%%
TP = 0;
tp = zeros(1,Seg_Data.Frame_Num);
NGT = 0;
ngt = zeros(1,Seg_Data.Frame_Num);
NS = 0;
ns = zeros(1,Seg_Data.Frame_Num);
%%
for t = 1:Seg_Data.Frame_Num
   G = imread(GT_Data.Frame_name{t});
   S = imread(Seg_Data.Frame_name{t});
   Gbw = G==1;
   Sbw = S(:,:,2)>0.5;
   sizeG = size(Gbw);
   sizeS = size(Sbw);
   sizeDiff = (sizeG-sizeS)/2;
   Gbw = Gbw(sizeDiff(1)+1:end-sizeDiff(1),sizeDiff(2)+1:end-sizeDiff(2));
   GL = bwlabel(Gbw);
   SL = bwlabel(Sbw);
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
           intersection = sum(Gl(:)&Sl(:));
           union = sum(Gl(:)|Sl(:));
           J = intersection./union;
           if J > 0.5
               TP = TP+1;
               tp(t) = tp(t)+1;
           end 
       end     
   end
end

Rec = TP./NGT;
Prec = TP./NS;
rec = tp./ngt;
prec = tp./ns;