%% Step_3_ComputeShapeMatching.m
%
% The script computes for all points in each mesh the corresponding points in a
% reference shape in the dataset. We use 5 different shape matching approaches
% based on the choices: ['FM', 'FM_opt', 'FM_net', 'ICP', 'CPD']
%
% Results are saved as N x 2 vectors containing the IDs of the points to be
% matched and the corresponding matching points.
%
%% Author: Filip Konstantinos <filip.k@ece.upatras.gr>
%% Last updated: 2021-02-03

% Structure list
structure_list = {'femur', 'femoral_cart', 'tibia', ...
                  'tibial_cart_med', 'tibial_cart_lat'};

% Folder containing all .stl files
dataset_dir = 'Data/OAI_ZIB/original/geometries';


% Input Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% select correspondence method
CorrespondenceMethod = 'FM'; % options: ['FM', 'FM_opt', 'FM_net', 'ICP', 'CPD']

k = 35; % number of eigenfunction used in the Functional Maps framework

% select to plot resutls
plot_results = true;

% rerence subject in dataset
reference_shape_tag = '9017909';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load trained FMnet IDs for each structures (used in FM_net method)
fid = fopen('Results\train_knee_flags\Net_IDs.txt');
if fid < 0, error('Cannot open file.'); end
CC = textscan(fid, '%s');
log_id = CC{1};
fclose(fid);

% ==============================================================================
% for each structure in the structure list...
for i = 1:length(structure_list)

    % get all the training shapes of the structure
    dataset = dir(sprintf(...
        strcat(dataset_dir, '/off_clean/%s/*.off'), structure_list{i}));

    % set the shape_X (reference) shape ID and name
    idx = find(contains({dataset.name}', reference_shape_tag));
    fname = dataset(1).name; fname(5:11) = reference_shape_tag;

    % setup struct to save results of correspondence
    clear labelmaps_list
    labelmaps_list = struct();

    % for each training shape...
    for idy = 1:length(dataset)

        fprintf('Model %d Part %d out of %d \n', [idy idx length(dataset)])

        % load shapes
        shape_X = loadoff(strcat(dataset(idx).folder, '/', dataset(idx).name));
        shape_Y = loadoff(strcat(dataset(idy).folder, '/', dataset(idy).name));

        % set file-name of saved data (eigenfun + descriptor)
        shape_X_evecs_fname = strrep( strcat(...
            dataset(idx).folder, '/evecs/', dataset(idx).name), '.off', '.mat');
        shape_X_desc_fname = strrep(strcat(...
            dataset(idx).folder, '/shot/', dataset(idx).name), '.off', '.mat');
        shape_Y_evecs_fname = strrep( strcat(...
            dataset(idy).folder, '/evecs/', dataset(idy).name), '.off', '.mat');
        shape_Y_desc_fname = strrep(strcat(...
            dataset(idy).folder, '/shot/', dataset(idy).name), '.off', '.mat');

        % load from file or compute eigenfunction
        [shape_X.evecs, shape_X.evecs_trans, shape_X.evals] = getEigenFunctions(...
            shape_X, k, shape_X_evecs_fname);
        [shape_Y.evecs, shape_Y.evecs_trans, shape_Y.evals] = getEigenFunctions(...
            shape_Y, k, shape_Y_evecs_fname);

        % load from file or compute descriptors
        shape_X.desc = getDescriptor(shape_X, 'SHOT', shape_X_desc_fname);
        shape_Y.desc = getDescriptor(shape_Y, 'SHOT', shape_Y_desc_fname);

        % solve correspondence problem
        switch CorrespondenceMethod
        case 'FM_partial'
            disp('Solve FM Partial...');

            % partial mapped to full
            B = shape_X.evecs_trans * normc([shape_X.desc]); % full
            A = shape_Y.evecs_trans * normc([shape_Y.desc]); % partial

            % estimated rank r
            est_rank = sum(diag(shape_Y.evals) - max(diag(shape_X.evals))<0);

            % manifold optimization problem
            problem.M = stiefelfactory(k, est_rank);
            lambda1 = shape_X.evals;
            mu = 1e+10; % weights
            W = 1 - diag(ones(est_rank,1)); % off diagonal
            Wr = [eye(est_rank) zeros(est_rank,k-est_rank)];

            % define problem (unknown = Q in the paper)
            problem.cost = @(X)mu*sum(sum((Wr*A - X'*B).^2)) + sum(sum(((X'*lambda1*X).*W).^2)) ;
            problem.egrad = @(X)(2*mu*(B*B'*X - B*(Wr*A)') + 4*(lambda1*X*X'*lambda1*X - (repmat(diag(X'*lambda1*X)',k,1)).*(lambda1*X)));
            checkgradient(problem);
            options.maxiter = 10000;
            options.verbosity = 0;
            x0 = [eye(est_rank);zeros(k-est_rank,est_rank)];
%             x0 = [diag(randr(-5, 5, est_rank));zeros(k-est_rank,est_rank)];

            % solve problem
            X_out = conjugategradient(problem,x0,options);

            % non rigid alignment in the r-dimensional eigenspace
            % (matches found are from full to partial)
            shape_Y_ = shape_Y; shape_Y_.evecs = shape_Y.evecs*X_out;
            shape_X_ = shape_X; shape_X_.evecs = shape_X.evecs*Wr';
%             [~, matches] = icp_refinement(shape_Y_, shape_X_, est_rank, [eye(est_rank)], 100); % full to partial
            [~, matches] = icp_refinement(shape_X_, shape_Y_, est_rank, [eye(est_rank)], 100); % partial to full
        case 'FM'
            % Orthogonal Procrustes Problem ->> min ||C*A-B|| s.t. C_t*C = I
            disp('Solve FM...');

            % descriptor projection onto the eigenfunctions A = Phi_t * F
            A = shape_X.evecs_trans * normc([shape_X.desc]);
            B = shape_Y.evecs_trans * normc([shape_Y.desc]);

            % Solve SVD
            [U, ~, V] = svd(B * A');
            C = U * V'; % C can be thought as a first alignment in the eigenspace

            % mesh registration in the eigenspace and point-to-point matches
            [C, matches] = icp_refinement(shape_Y, shape_X, k, C, 100);

        case 'FM_opt'
            % Orthogonal Procrustes Problem ->> min ( ||C*A-B|| +
            % a * trace( C * LambdaA * C_t) ) s.t. C_t*C = I
            disp('Solve FM Optimization...');

            A = shape_X.evecs_trans * normc([shape_X.desc]);
            B = shape_Y.evecs_trans * normc([shape_Y.desc]);

            % set optimization problem...
            a = 1e-4; % weight of the regularization term
            lambda_M = shape_X.evals;
            problem.M = stiefelfactory(k, k);
            problem.cost = @(X)sum(sum((A' * X - B').^2)) + ...
                a * trace((X * lambda_M * X'));
            problem.egrad = @(X)2 * A * (A' * X - B') + ...
                a * X * (lambda_M + lambda_M');

            % ...and solve it!
            options.maxiter = 10000;
            options.verbosity = 0;
            x0 = eye(k);
            C = conjugategradient(problem, x0, options);
            C = C';

            % refine resutls and get point-to-point correspondence
            [C, matches] = icp_refinement(shape_Y, shape_X, k, C, 100);

        case 'FM_net'
            disp('Solve FM_Net...');
            k = k;
            if (i == 4) % reload eigenfunctions for 'tibial-cart-med'
                k = 20;
                [shape_X.evecs, shape_X.evecs_trans, shape_X.evals] = getEigenFunctions(...
                    shape_X, k, shape_X_evecs_fname);
                [shape_Y.evecs, shape_Y.evecs_trans, shape_Y.evals] = getEigenFunctions(...
                    shape_Y, k, shape_Y_evecs_fname);
            end

            % load matrix C from file
            C_fname = sprintf('./Results/functional_maps_%s.mat', structure_list{i});
            C_est = struct2cell(load(C_fname));
            C = squeeze(C_est{idy});

            % refine results and compute point-to-point matches
            [C, matches] = icp_refinement(shape_Y, shape_X, k, C, 100);

        case 'ICP'
            % Iterative transformation and closest point match
            disp('Solve ICP...');

            [R, t, BRt, ~, ~, ~, matches, ~, ~, ~] = icp(...
                shape_Y.VERT, shape_X.VERT, 'MaxIter', 5000);

        case 'CPD'
            disp('Solve CPD...');

            % Note: Non-rigid methods do not work...
            opt.method = 'affine'; % ['rigid', 'affine', 'nonrigid', 'nonrigid_lowrank']
            opt.corresp = 1;
            opt.viz = 0;
            opt.fgt = 2;
            opt.outliers = 0.2;
            opt.beta = 1;
            opt.lambda = 5;

            [~, matches] = cpd_register(shape_Y.VERT, shape_X.VERT, opt);

        end

        % append results to struct for each training shape
        labelmaps_list.(strrep(dataset(idy).name, '.off', '')).(...
            strrep(dataset(idx).name, '.off', '')) = matches;

        % plot results
        if plot_results == true
            colors = create_colormap(shape_Y, shape_Y);
            figure; subplot(121); colormap(colors);
            plot_scalar_map(shape_Y, [1:size(shape_Y.VERT, 1)]'); freeze_colors;
            title(sprintf('Y shape (# Vertices: %d)', [size(shape_Y.VERT, 1)]));
            axis off; hold on
            subplot(122); colormap(colors(matches, :));
            plot_scalar_map(shape_X, [1:size(shape_X.VERT, 1)]'); freeze_colors;
            title(sprintf('X shape (# Vertices: %d)', [size(shape_X.VERT, 1)]));
            axis off; hold off

            if (ismember({'FM', 'FM_opt', 'FM_net'}, CorrespondenceMethod))
                figure; imagesc(C); title('Functional Correspondence Matrix');
                axis image; colormap(jet); colorbar; axis off;
            end
            drawnow
        end

        % save correspondences to file
        save_path = strcat(dataset(idy).folder, '/ground_truth_', CorrespondenceMethod);
        if 7 ~= exist(save_path)
            mkdir(save_path)
        end
        labelmaps_list_temp = struct2cell(labelmaps_list);
        labels = labelmaps_list_temp{idy};
        mat_fname = strrep(dataset(idy).name, '.off', '.mat');
        save(strrep(strcat(save_path, '/', mat_fname), '\', '/'), 'labels');

    end
end

disp('END')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [r] = randr(a, b, size)
  % Help function for computing a list of random values in range (a,b)
    r = (b-a).*rand(size,1) + a;
end
