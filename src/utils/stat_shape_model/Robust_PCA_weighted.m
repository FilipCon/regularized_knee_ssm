function [L, S, x_mean, Evalues, Evectors, ssmV] = Robust_PCA_weighted(X,w, lambda, mu, tol, max_iter)
    % - X is a data matrix (of the size N x M) to be decomposed
    %   X can also contain NaN's for unobserved values
    % - lambda - regularization parameter, default = 1/sqrt(max(N,M))
    % - mu - the augmented lagrangian parameter, default = 10*lambda
    % - tol - reconstruction error tolerance, default = 1e-6
    % - max_iter - maximum number of iterations, default = 1000
    W = spdiags(w, 0, size(w,1), size(w,1));
X=W*X;

    [M, N] = size(X);
    unobserved = isnan(X);
    X(unobserved) = 0;
    normX = norm(X, 'fro');

    x_mean=sum(X,2)/N;
    x=(X-repmat(x_mean,1,N))/ sqrt(N-1);

    % default arguments
    if nargin < 3
        lambda = 1 / sqrt(max(M,N));
    end
    if nargin < 4
        mu = 10*lambda;
    end
    if nargin < 5
        tol = 1e-6;
    end
    if nargin < 6
        max_iter = 2500;
    end

    % initial solution
    L = zeros(M, N);
    S = zeros(M, N);
    Y = zeros(M, N);

    for iter = (1:max_iter)
        % ADMM step: update L and S
        [L, ~,~] = Do(1/mu, x - S + (1/mu)*Y);
        S = So(lambda/mu, x - L + (1/mu)*Y);
        % and augmented lagrangian multiplier
        Z = x - L - S;
        Z(unobserved) = 0; % skip missing values
        Y = Y + mu*Z;

        err = norm(Z, 'fro') / normX;
        if (iter == 1) || (mod(iter, 10) == 0) || (err < tol)
            fprintf(1, 'iter: %04d\terr: %f\trank(L): %d\tcard(S): %d\n', ...
                    iter, err, rank(L), nnz(S(~unobserved)));
        end
        if (err < tol) break; end
    end

    [U2,S2] = svd(L,0);
    Evalues=diag(S2).^2;
    Evectors=bsxfun(@times,U2,sign(U2(1,:)));

    Eval2=sqrt(Evalues);

    for i=1:length(Evalues)
        ssmV(:,i)=Eval2(i,1)*Evectors(:,i);

    end

%     PCcum=cumsum(Evalues)./sum(Evalues);

    % Remove weighting from mean
x_mean= spdiags(1./w, 0, size(w,1), size(w,1)) * x_mean;
ssmV = spdiags(1./w, 0, size(w,1), size(w,1)) * ssmV;
Evectors = spdiags(1./w, 0, size(w,1), size(w,1)) * Evectors;

end

function r = So(tau, X)
    % shrinkage operator
    r = sign(X) .* max(abs(X) - tau, 0);
end

function [r, U, S] = Do(tau, X)
    % shrinkage operator for singular values
    [U, S, V] = svd(X, 'econ');
    r = U*So(tau, S)*V';
end