%% Step_4_MultiOrganSSM.m
%
% Create a multi-structure Statistical Shape Model based on the SOMOS framework.
% Modeling is either on the Canonical Corelation Analysis (whcih gives the
% corelation of each structure), either as the statndard PDM for multiple
% srtuctures.
%
%% Author: Filip Konstantinos <filip.k@ece.upatras.gr>
%% Last updated: 2021-02-03

clear all; clc
% Structure list
structure_list_full = {'femur', 'femoral_cart', 'tibia', ...
                       'tibial_cart_med', 'tibial_cart_lat'};

% rerence subject
reference_shape_tag = '9017909';

% Folder containing all .stl files
dataset_dir = 'Data/OAI_ZIB/original/geometries';

% Input parameteres
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options.verbose = false; % If verbose is true plot shape variation of a single mode
use_RPCA = true; % if true use robust-pca, else use the classical PCA
root_structure = 5; % in ssm using cca one structure has weight = 1; if 0, cca is not used.
method = "FMNet"; % load files with name extensions equal to the method used to compute correspondences

% Merge training shapes in a training_data structure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Constructing Training Data')

% load tags of training shapes.
fid = fopen('Data\OAI_ZIB\training_set.txt');
if fid < 0, error('Cannot open file.'); end
CC = textscan(fid, '%s');
training_subjects = CC{1};
fclose(fid);

% setup struct to save training training_data
training_data = struct;

% for each training subject...
for idx = 1:numel(training_subjects)

    % setup the fields of the struct
    training_data(idx).vertices = [];
    training_data(idx).faces = [];
    training_data(1).numverts = [];
    training_data(1).numfaces = [];

    % for each structure...
    for ids = 1:length(structure_list_full)

        % get a list of all shape file names if dir
        dataset = dir(sprintf(...
            strcat(dataset_dir, '/off_clean/%s/*.off'), structure_list_full{ids}));

        % get a list of all files containing the computed matches
        gt_datapath = dir([dataset(1).folder sprintf('/ground_truth_%s/*.mat', method)]);

        % load reference shape
        clear shape_X
        fname = dataset(1).name; fname(5:11) = reference_shape_tag;
        shape_X = loadoff(strcat(dataset(1).folder, '/', fname));

        % load training shape
        clear shape_Y
        fname = dataset(1).name; fname(5:11) = training_subjects{idx};
        shape_Y = loadoff(strcat(dataset(1).folder, '/', fname));

        % load matches (X -> Y)
        fname = gt_datapath(1).name; fname(5:11) = training_subjects{idx};
        model_match = load(strcat(gt_datapath(1).folder, '/', fname));
        temp_cell = struct2cell(model_match.labels);
        matches = temp_cell{1};

        % append the registered trainingg shape to training_data structure
        training_data(idx).vertices = vertcat(training_data(idx).vertices,...
             shape_Y.VERT(matches, :)); % matched vertices of Y...
        training_data(ids).faces = shape_X.TRIV; % ...with connectivity of X!!!
        training_data(ids).numverts = length(shape_X.VERT);
        training_data(ids).numfaces = length(shape_X.TRIV);
    end
end

% align shapes all registered trainign shapes
training_data_aligned = alignTrainingData(training_data);

% perform (or not) canonical corelation analysis (CCA) and get weight matrix
% if root_structure = 0, the weight is the identity matrix
weights = computePCAWeigts(training_data_aligned, root_structure);

% PCA
%===============================================================================
disp('Calculating PCA...');

% vectorize aligned training shapes
num_verts = size(training_data_aligned(1).vertices, 1);
num_training_shapes = length(training_data);
training_data_vectorized = zeros(num_verts * 3, num_training_shapes);
for i = 1:num_training_shapes
    training_data_vectorized(:, i) = ...
        [training_data_aligned(i).vertices_aligned(:, 1); ...
         training_data_aligned(i).vertices_aligned(:, 2); ...
         training_data_aligned(i).vertices_aligned(:, 3)];
end

% solve pca/rpca
if use_RPCA == true
    [~, ~, mean_shape_vectorized, eig_values, eig_vectors_normalized, eig_vectors_scaled] = ...
        Robust_PCA_weighted(training_data_vectorized, weights);
else
    [mean_shape_vectorized, eig_values, eig_vectors_normalized, eig_vectors_scaled] = ...
        PCA_weighted(training_data_vectorized, weights);
end

% Keep the eigen-vectors/values that explain the 98% variance
i = find(cumsum(eig_values) > sum(eig_values) * 0.98, 1, 'first');
eig_vectors_normalized = eig_vectors_normalized(:, 1:i); eig_values = eig_values(1:i);
eig_vectors_scaled = eig_vectors_scaled(:, 1:i); eig_values = eig_values(1:i);

% reshape mean shape
mean_shape = unvectorize_shape(mean_shape_vectorized);

% setup SSM structure
ssm.eig_values = eig_values;
ssm.eig_vectors_normalized = eig_vectors_normalized;
ssm.eig_vectors_scaled = eig_vectors_scaled;
ssm.mean_shape = mean_shape;
ssm.mean_shape_vectorized = mean_shape_vectorized;
ssm.faces = {training_data_aligned.faces};
ssm.numverts = {training_data_aligned.numverts};
ssm.weights = weights;
% ssm.training_data_vectorized = training_data_vectorized;

%===============================================================================
% store resutls
if use_RPCA == true
    postfix = "_RPCA";
else
    postfix = "_PCA";
end

if root_structure ~= 0
    save_path = strcat(sprintf('Results/stat_shape/MultiOrganSSM_%s_', ...
        structure_list_full{root_structure}), method, postfix);
else
    save_path = strcat('Results/stat_shape/MultiOrganSSM_', method, postfix);
end

if 7 ~= exist(save_path)
    mkdir(save_path)
end

full_path = strcat(save_path, '/', 'training_data_aligned.mat');
save(strrep(full_path, '\', '/'), 'training_data_aligned');

full_path = strcat(save_path, '/', 'ssm.mat');
save(strrep(full_path, '\', '/'), 'ssm');

disp('End of calculation')

%===============================================================================
%% animate the variations of principal components
% 
% close all; figure;
% pc_comp = 1; % select principal component (mode) of pca
% 
% 
% if options.verbose == true
%     % change the eigenvalue of that mode...
%     for i = -3 * sqrt(ssm.eig_values(pc_comp)): ... % start
%             0.1:...                                       % step
%             3 * sqrt(ssm.eig_values(pc_comp))       % end
% 
%         % .. setup the matrix that multiplies the eig_value
%         A = zeros(length(ssm.eig_values), 1);
%         A(pc_comp) = i; A = diag(A);
% 
%         % ...and plot the deformed ssm
%         plotshapemode(length(ssm.eig_values), ...
%             training_data_aligned, ssm.eig_vectors_scaled, ssm.mean_shape_vectorized, A,...
%                 length(structure_list_full));
%         view([0, 0]); rotate3d on; drawnow; pause(0.01)
%     end
% end

disp('End of calculation')
