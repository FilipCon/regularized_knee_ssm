%% Step_5_ApplySSM.m
%
% The script fits the built SSMs to new shapes in the testing set based on the
% optimization scheme that searches for a solution close to new shape and
% previously observed shapes.
%
%% Author: Filip Konstantinos <filip.k@ece.upatras.gr>
%% Last updated: 2021-02-03

% Structure list
structure_list_full = {'femur', 'femoral_cart', 'tibia', ...
                       'tibial_cart_med', 'tibial_cart_lat'};

% Folder containing all .stl files
dataset_dir = 'Data/OAI_ZIB/original/geometries';

% labelmap dataset
labelmaps_dataset = dir('Data/OAI_ZIB/original/labelmaps/*.nii.gz');

%===============================================================================
% input parameters
%===============================================================================

structure_list_partial = {'femur',  'femoral_cart', 'tibia', ...
  'tibial_cart_med'};

root_structure = 5; % idt in structure_list_full
regularization_weight = 1e-10;
% regularization_weight = 0;

method = "FMNet"; % file extension of the method used
pca_method = "_RPCA"; % file extension of the method used

%===============================================================================
% Data Preparation
%===============================================================================
% load tags of testing subjects
fid = fopen('Data\OAI_ZIB\testing_set.txt');
if fid < 0, error('Cannot open file.'); end
CC = textscan(fid, '%s');
testing_subjects = CC{1};
fclose(fid);

% create struct with all testing subjects (data:<structure>:<subject>)
testing_data = struct;
for i = 1:length(structure_list_full)
    for j = 1:numel(testing_subjects)
        dataset = dir(sprintf(...
            strcat(dataset_dir, '/off_clean/%s/*.off'), structure_list_full{i}));
        fname = dataset(i).name; fname(5:11) = testing_subjects{j};
        testing_data.(structure_list_full{i}).(strcat('oai_',testing_subjects{j})) = loadoff(...
            strcat(dataset(1).folder, '/', fname));
    end
end

% load ssm
if root_structure == 0
    dir_name = strcat('Results\stat_shape\MultiOrganSSM_', method);
else
    dir_name = strcat(sprintf('Results/stat_shape/MultiOrganSSM_%s_', ...
    structure_list_full{root_structure}), method, pca_method);
end
load(strcat(dir_name, '\ssm.mat'));

% setup pointers and get faces for each structure in ssm
[mean_full_begin_ptr, mean_full_end_ptr] = computeRegionPointers(ssm.numverts);
ssm_faces = ssm.faces;
ssm_faces(cellfun('isempty', ssm_faces)) = [];

%===============================================================================
% Begin
%===============================================================================
% metrics result file
metric_file_id = fopen(strcat('Results/evaluation_metrics_', structure_list_full{root_structure}, ...
    '_', method, pca_method, '_', num2str(regularization_weight), '.txt'), 'w');

for idt = 1:numel(testing_subjects)

    test_subject_tag = testing_subjects{idt};
    fprintf('Subject %s : %d out of %d  \n', test_subject_tag, idt, numel(testing_subjects))
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    % set missing structures
    [ssm_partial, target_shape] = setMissingStructures(...
        testing_data, test_subject_tag, ssm, ...
        structure_list_full, structure_list_partial);

    % apply the shape fitting algorithm to partial target
    [ssm_fit_partial, ssm_fit_full, modes] = SSMfitter_ICP(...
        target_shape, ssm_partial, ssm, regularization_weight);

    % rescale (initial volume were in mm)
    ssm_fit_full = ssm_fit_full * 100;


    % Evaluation Metrics
    %=============================================================================
    %=============================================================================

    % load target labelmap
    ref_labelmap = load_untouch_nii(strcat(labelmaps_dataset(1).folder, '/', ...
         test_subject_tag, '.nii.gz'));

    % init empty volume
    Volume = zeros(size(ref_labelmap.img));

    for n = 1:numel(structure_list_full)
        % test shape
        target = testing_data.(structure_list_full{n}).(strcat('oai_', test_subject_tag));
        target.VERT = target.VERT * 100;

        % fit ssm
        surface = {};
        surface.vertices = ssm_fit_full(mean_full_begin_ptr(n):mean_full_end_ptr(n), :);
        surface.faces = cell2mat(ssm_faces(n));

        % tranform structure to labelmap
        fit_shape_nii = mesh2label_map(surface, ref_labelmap, n);
        Volume = Volume + double(fit_shape_nii.img);
        Volume(Volume >= n + 1) = n;

        % volume metrics
        Dice = computeDiceScore(ref_labelmap.img == n, fit_shape_nii.img == n);
        VOE = computeVolumeOverlap(ref_labelmap.img == n, fit_shape_nii.img == n);
        VD = computeVolumeDifference(ref_labelmap.img == n, fit_shape_nii.img == n);

        % distance metrics
        [Hausdorff, d1, d2] = hausdorff_percentile(surface.vertices, target.VERT, 100);
        Hausdorff95 = hausdorff_percentile(surface.vertices, target.VERT, 95);
        ASD = mean([d1 d2]);
        RSD = sqrt(mean([d1 d2].^2));

        disp(sprintf('Dice of %s  = %f percent', structure_list_full{n}, Dice));
        fprintf(metric_file_id, '%f, %f, %f, %f, %f, %f, %f, ', ...
            Dice, VOE, VD, Hausdorff, Hausdorff95, ASD, RSD);
    end

    fprintf(metric_file_id, '\n');

    % save fit ssm as labelmap
    fit_shape_nii.img = Volume;
    save_untouch_nii(fit_shape_nii, ...
        strrep([pwd '/Results/stat_shape/SSM_labels.nii.gz'], '\', '/'));
end

fclose(metric_file_id);

disp('END')

