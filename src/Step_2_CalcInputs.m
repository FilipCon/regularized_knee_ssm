%% Step_2_CalcInputs.m
%
% In this step we calculate the required data used for the Deep Functional Maps
% framework. Specifically, for all meshes in folder, we compute the following:
% 1) Geodesic distances
% 2) Eigenfuncitons
% 3) SHOT descriptor
%
% All results are saved in the data directory containing the geometries.
%
%% Author: Filip Konstantinos <filip.k@ece.upatras.gr>
%% Last updated: 2021-02-03

% Structure structure_list
structure_list = {'femur', 'femoral_cart', 'tibia', ...
                  'tibial_cart_med', 'tibial_cart_lat'};

% Folder containing all .stl files
dataset_dir = 'Data/OAI_ZIB/original/geometries';

%% Calculate the geodesic distances for all point in each mesh using the Fast
% marching algorithm.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% for each structure
for i = 1:length(structure_list)

    % list of all .off files in data folder
    dataset = dir(sprintf(...
        strcat(dataset_dir, '/off_clean/%s/*.off' ), structure_list{i}));

    disp('Computing Distance Maps...')

    % for each .off file
    for idx = 1:length(dataset)

      % Compute the distance map
        fprintf('------ Model %d out of %d -------- \n', [idx length(dataset)])
        D = computeDistanceMaps(strcat(dataset(idx).folder, '/', dataset(idx).name));

        % Path to save the file. If does not exist, create subdir within the
        % location dir of .off files
        save_path = strcat(dataset(idx).folder, '/', 'distance_maps/');
        if 7 ~= exist(save_path)
            mkdir(save_path)
        end

        % name of file and full path
        mat_fname = strrep(dataset(idx).name, '.off', '.mat');
        full_path = strcat(save_path, '/', mat_fname);

        save(strrep(full_path, '\', '/'), 'D');
    end

    disp('End of calculation')

%% Calculate Eigenfunction
% (Compute a total of k=100 eigenfunctions)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    disp('Computing Eigenfunctions...')
    % for each .off file
    for idx = 1:length(dataset)

        fprintf('------ Model %d out of %d -------- \n', [idx length(dataset)])
        % Load .off file
        shape = loadoff(strcat(dataset(idx).folder, '/', ...
                              dataset(idx).name));

        k = 100; % k first eigenfunctions
        [evecs, evecs_trans, evals] = calc_eigenfun(shape, k);

        save_path = strcat(dataset(idx).folder, '/evecs');
        if 7 ~= exist(save_path)
            mkdir(save_path)
        end

        mat_fname = strrep(dataset(idx).name, '.off', '.mat');
        full_path = strcat(save_path, '/', mat_fname);
        save(strrep(full_path, '\', '/'), 'evecs', 'evecs_trans', 'evals');
    end

% Calculate SHOT descriptor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Computing SHOT descriptor...')

    for idx = 1:length(dataset)

        fprintf('------ Model %d out of %d -------- \n', [idx length(dataset)])
        % Load .off file
        shape = loadoff(strcat(dataset(idx).folder, '/', dataset(idx).name));

        desc = getDescriptor(shape, 'SHOT');

        % Save eigenfunctions and SHOT to .mat
        save_path = strcat(dataset(idx).folder, '/shot');
        if 7 ~= exist(save_path)
            mkdir(save_path)
        end

        mat_fname = strrep(dataset(idx).name, '.off', '.mat');
        full_path = strcat(save_path, '/', mat_fname);
        save(strrep(full_path, '\', '/'), 'desc');
    end
end

disp('End of calculation');


% ==============================================================================
function [D] = computeDistanceMaps(shape_fname)
    % Load .off file
    surface = loadoff(shape_fname);
    % number of vertices
    N = size(surface.VERT, 1);
    % Compute Geodesic Distances
    D = zeros(N, N); % Memory allocation
    D = fastmarch(surface.TRIV, surface.VERT(:, 1), surface.VERT(:, 2), ...
        surface.VERT(:, 3), [1:N]);
    D = single(D);
end
