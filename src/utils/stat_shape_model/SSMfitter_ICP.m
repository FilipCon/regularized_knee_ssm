function [ssm_fit_partial, ssm_fit_full, modes] = SSMfitter_ICP(...
    target_shape, ssm_partial, ssm, w)
% Shape fitting algorithm proposed by Cootes et al., modified for multi-structure SSMs.

    % begin-end pointers for each structure in multi-shape
    [X_begin, X_end] = computeRegionPointers(ssm_partial.numverts);
    [Y_begin, Y_end] = computeRegionPointers(target_shape.numverts);

    % initial alignment of ssm
    [R1,t1, ssm_partial_aligned] = icp(target_shape.vertices, ssm_partial.vertices);

    % init state
    index = 2;
    error = [0; 1000000];
    modes = zeros(size(ssm_partial.eig_vectors_scaled, 2), 1);
    ssm_fit_partial = ssm_partial_aligned;
    target_aligned = target_shape.vertices;

    % fit until error is smaller than tolerance
    while (abs(error(index, :) - error(index - 1, :))) > 0.00001
        [error(index + 1, :), target_aligned, ssm_fit_partial, modes] = ssm_fitter_itr(...
            target_aligned, ssm_partial_aligned, ssm_fit_partial,...
            ssm_partial.eig_vectors_scaled, ssm.eig_values, w, ...
            modes, target_shape.numverts, X_begin, X_end, Y_begin, Y_end);

        index = index + 1;
    end

    % alignment of smm
    [R2, t2, ssm_fit_partial] = icp(target_shape.vertices, ssm_fit_partial);
    mean_shape_full_aligned = (ssm.mean_shape * R1 + t1) * R2 + t2;

    % apply pca parameters to full-structure SSM
    ssm_fit_full = unvectorize_shape(vectorize_shape(mean_shape_full_aligned) + ...
        ssm.eig_vectors_scaled * modes);

end
%===============================================================================

function [error, Y_aligned, X_est, modes] = ssm_fitter_itr(...
    Y, X, X_est, evecs, evals, w,...
    init_modes, numverts_x, X_begin, X_end, Y_begin, Y_end)

    % find knn in multi-structure shapes (both directions)
    [KAB, KBA] = multiStructureKNN(X_est, Y, numverts_x, ...
        X_begin, X_end, Y_begin, Y_end);

    % create augmented shapes
    X_aug = vertcat(X,         X(KBA, :));
    Y_aug = vertcat(Y(KAB, :), Y        );

    % set target shape for fitting
    Z_aug = vectorize_shape(Y_aug - X_aug);

    % get columns x, y, z of eigenvectors...
    s = size(evecs) / 3;
    BTX = evecs(1 : s, :);
    BTY = evecs(s + 1 : 2 * s, :);
    BTZ = evecs(2 * s + 1 : end, :);

    % ... and create the augmented version as well
    BTX_aug = vertcat(BTX, BTX(KBA,:), BTY, BTY(KBA,:), BTZ, BTZ(KBA,:));

    if w == 0 % depending on regularization term...
        % ... solve analytically
        modes = pinv(BTX_aug) * Z_aug;
    else
        % ...  or solve as optimization (non-linear least square problem)
        options = optimoptions(@lsqnonlin, 'Algorithm', 'levenberg-marquardt',...
            'MaxFunctionEvaluations', 100 * size(BTX_aug, 2), 'MaxIterations', 1000,...
            'Display', 'off');

        c = sqrt(((2 * pi)^length(evals)) * det(diag(evals)));
        fun = @(x)(norm(BTX_aug * x - Z_aug))^2 + w * c * exp(x' * (diag(evals) \ x) / 2);
        modes = lsqnonlin(fun, init_modes, [], [], options);
    end

    % create the reconstructed (fit) ssm with current estimate modes
    X_est = unvectorize_shape(vectorize_shape(X) + evecs * modes);

    % re-evaluate knn with the fit ssm
    [KAB, KBA] = multiStructureKNN(X_est, Y, numverts_x, ...
        X_begin, X_end, Y_begin, Y_end);

    % re-new augmented shapes...
    X_aug = vertcat(X_est,     X_est(KBA, :));
    Y_aug = vertcat(Y(KAB, :), Y);

    % ... perform rigid transformation of Y_aug to X_aug, compute distance error, ...
    [error, Y_aug_aligned] = procrustes(X_aug, Y_aug, 'scaling', 0);

    % .. and return the aligned shape Y (the ORIGINAL!)
    Y_aligned = Y_aug_aligned(size(KAB, 1) + 1 : end, :);
end
%===============================================================================

function [KAB, KBA] = multiStructureKNN(X, Y, numverts_x, X_begin, X_end, Y_begin, Y_end)
% knn search in multi-structure shapes.
% X, Y are shapes [N x 3] containing stacked structures.
% X_begin, X_end and Y_begin, Y_end are pointers to rows of X, Y, respectively,
% pointing each structure in the matrices.

    KAB = []; KBA = [];
    for i = 1 : numel(numverts_x)
        tmp_KAB = knnsearch(Y(Y_begin(i) : Y_end(i), :), X(X_begin(i) : X_end(i), :));
        tmp_KBA = knnsearch(X(X_begin(i) : X_end(i), :), Y(Y_begin(i) : Y_end(i), :));
        KAB = [KAB; tmp_KAB + Y_begin(i) - 1];
        KBA = [KBA; tmp_KBA + X_begin(i) - 1];
    end
end