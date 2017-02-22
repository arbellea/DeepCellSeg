%%
GT_dir = '';
GT_txt = '.png';
GT_exp = '(\t+)';
GT_Data = Load_Data(GT_dir,GT_txt,GT_exp);
%%
Seg_dir = '';
Seg_txt = '.png';
Seg_exp = '(\t+)';
Seg_Data = Load_Data(Seg_dir,Seg_txt,Seg_exp);
%%
TP = 0;
tp = zeros(1,Seg_Data.Frame_Num);
NGT = 0;
ngt = zeros(1,Seg_Data.Frame_Num);
NS = 0;
ns = zeros(1,Seg_Data.Frame_Num);

for t = 1:Seg_Data.Frame_Num
   G = imread(GT_Data.Frame_name{t});
   S = imread(Seg_Data.Frame_name{t});
   Gbw = G(:,:,2)>0.5;
   Sbw = S(:,:,2)>0.5;
   
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