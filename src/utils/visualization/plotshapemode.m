function plotshapemode(nr_of_mode,Data,ssmV,MEAN,A,  numshapes)

% function to plot principal components of shape variation at the +3 standard
% deviation, average and -3 SD. 

%INPUT
% nr_of_mode : total nr of shape mode to plot
% ssmV : set of principal components of SSM
% MEAN: mean shape vector
% Fdata : triangulation matrix of training data 
% A = select 1 mode to vary 

% %collumn of ssmV gives nth principal component
% 
PC=MEAN+sum(ssmV(:,1:nr_of_mode)*A,2);
PCplus = reshape(PC, size(PC,1)/3, 3);

numverts = {};
for i=1:numshapes
    numverts{i} = Data(i).numverts ;
end

n=2;
point1(1) = 1;
point2(1) = numverts{1};
while n <= numel(numverts)
    
    point1(n) = point2(n-1)+1;
    point2(n) = Data(n).numverts + point2(n-1);    
    
    n = n+1;
end
n=1; 
while n <= numel(numverts)
    clear surface
    surface.VERT = [PCplus(point1(n):point2(n),1) PCplus(point1(n):point2(n),2) PCplus(point1(n):point2(n),3)] ;
    surface.TRIV = Data(n).faces;
    color = ones(size(surface.VERT ,1),1)+n;
    showshape(surface, color); hold on
    n=n+1;
end
hold off
lighting gouraud
camlight headlight


