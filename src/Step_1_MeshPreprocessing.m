%% Step_1_MeshPreprocessing.m
%
% This script is used to apply a series of filters using Meshlabserver on the
%.stl files resulted from Cube matching algorithm. For that reason you need to
%have Meshlab installed, and the Meshlabserver executable in your system's PATH.
% Processed files are saved in .off format.

% Filters used are found in the "./src/utils/mesh_preproc/mesh_preproc.mlx"
% file. Namely the filters applied are
% 1) Merge Close Vertices
% 2) Remove Duplicate Vertices
% 3) Remove Duplicate Faces
% 4) Remove Isolated pieces
% 5) Simplification: Quadric Edge Collapse Decimation
% 6) Scale
% 7) Save as .off files
%
%% Author: Filip Konstantinos <filip.k@ece.upatras.gr>
%% Last updated: 2021-02-03

clear all; close all; clc
addpath(genpath('src'))

% Input data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Structure name list
structure_list = {'femur', 'femoral_cart', 'tibia', ...
                  'tibial_cart_med', 'tibial_cart_lat'};

% Folder containing all .stl files
dataset_dir = 'Data/OAI_ZIB/original/geometries';

% path to .mlx file
mlx_file = strrep([pwd ...
                     '/src/utils/mesh_preproc/mesh_preproc.mlx'], '\', '/');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n = 1:length(structure_list)
    dataset = dir(sprintf(strcat(dataset_dir, ...
        '/stl_original/%s/*.stl'), structure_list{n}));
    output_dir = sprintf(strcat(dataset_dir, '/off_clean/%s'), structure_list{n});

    % make dir
    if 7 ~= exist(output_dir)
        mkdir(output_dir)
    end

    for k = 1:length(dataset)

        fprintf('List %d : Model %d out of %d \n', [n k length(dataset)])

        % RUN MESHLAB SERVER
        formatSpec = sprintf('meshlabserver.exe -i %s -o %s -m fq wt -s %s', ...
            strrep(strcat(dataset(k).folder, '/', dataset(k).name), '\', '/'), ...
            strrep(strrep(strcat(pwd, '/', output_dir, '/', dataset(k).name), ...
            '.stl', '.off'), '\', '/'), mlx_file);
        system(formatSpec)

    end
end

disp('END')
