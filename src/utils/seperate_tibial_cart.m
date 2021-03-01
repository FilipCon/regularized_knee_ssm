%% Separates the tibial cartilage in TCM and TCL in the multi-label mask.
clc;clear all;close all
addpath(genpath('./src'))

labelmaps = dir('Data\OAI_ZIB\labelmaps/9269383.nii.gz');
list = [];
% for idx =1:length(labelmaps)
for idx = 1
labelmap_nii = load_untouch_nii([labelmaps(idx).folder '/' labelmaps(idx).name]) ;

V = labelmap_nii.img==4;
CC = bwconncomp(V);
S = regionprops(CC,'Centroid' );


X = [];
numofcomp = CC.NumObjects;
for i=1:numofcomp
    X = vertcat(X,S(i).Centroid(2:3));
end
[id, C] = kmeans(X,2);


if C(1)>C(2)
    for i_comp=1:numofcomp
        if id(i_comp)== 1
            V(CC.PixelIdxList{i_comp}) = 0; % keep lateral
        end
    end
else
    for i_comp=1:numofcomp
        if id(i_comp)== 2
            V(CC.PixelIdxList{i_comp}) = 0; % keep lateral
        end
    end
end
labelmap_nii.img(V) = 5;


save_untouch_nii(labelmap_nii,strrep([pwd '/Data/9269383.nii.gz'],'\','/'))



end

disp('end')