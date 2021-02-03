function [D, matches] = icp_refinement(M, N, r, C_init, max_iters)

% fprintf('Running ICP...\n');

X = N.evecs(:,1:r)';
Y = M.evecs(:,1:r)';

tree = createns(Y','nsmethod','kdtree');
matches = knnsearch(tree,[C_init*X]','K',1);

err = sum( sqrt(sum((C_init*X - Y(:,matches)).^2)) );
err = err / (size(X,2)*size(C_init,1));

% fprintf('(0) MSE: %.2e\n', err);

if max_iters == 0
    D = C_init;
    return
end


D_prev = C_init;
err_prev = err;

for i=1:max_iters

[U,~,V] = svd(X * Y(:,matches)');
    D = U * V(:,1:r)';
    D = D';

    matches = knnsearch(tree, [D*X]','K',1);

    err = sum( sqrt(sum((D*X - Y(:,matches)).^2)) );
    err = err / (size(X,2)*size(C_init,1));

%     fprintf('(%d) MSE: %.2e\n', i, err);

    if err > err_prev
%         fprintf('Local optimum reached.\n');
        D = D_prev;
        break;
    end

    if (err_prev - err) < 5e-6
%         fprintf('Local optimum reached.\n');
        break;
    end

    err_prev = err;
    D_prev = D;

end

end

