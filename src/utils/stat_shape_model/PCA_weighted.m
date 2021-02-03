function [x_mean, Evalues, Evectors, ssmV]=PCA_weighted(x, w)
% PCA using Single Value Decomposition
% Obtaining mean vector, eigenvectors and eigenvalues
%
% [Evalues, Evectors, x_mean]=PCA(x);
%
% inputs,
%   X : M x N matrix with M the trainingvector length and N the number
%              of training data sets
%
% outputs,
%   Evalues : The eigen values of the data
%   Evector : The eigen vectors of the data
%   x_mean : The mean training vector
%
%
W = spdiags(w, 0, size(w,1), size(w,1));

x=W*x;

s=size(x,2);
% Calculate the mean
x_mean=sum(x,2)/s;

% Substract the mean
x2=(x-repmat(x_mean,1,s))/ sqrt(s-1);

% Do the SVD
%[U2,S2] = svds(x2,s);
[U,S] = svd(x2,0);

Evalues=diag(S).^2;
Evectors=bsxfun(@times,U,sign(U(1,:)));

Eval2=sqrt(Evalues);

for i=1:length(Evalues)
    ssmV(:,i)=Eval2(i,1)*Evectors(:,i);

end
% PCcum=cumsum(Evalues)./sum(Evalues);


% Remove weighting
x_mean= spdiags(1./w, 0, size(w,1), size(w,1)) * x_mean;
ssmV = spdiags(1./w, 0, size(w,1), size(w,1)) * ssmV;
Evectors = spdiags(1./w, 0, size(w,1), size(w,1)) * Evectors;