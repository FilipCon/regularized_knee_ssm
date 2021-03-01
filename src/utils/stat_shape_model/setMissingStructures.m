function [ssm_partial, test_shape] = setMissingStructures(...
    testing_data, target_subject_tag, ssm, ...
    structure_list_full, structure_list_partial)

    % INIT
    [begin_ptr, end_ptr]  = computeRegionPointers(ssm.numverts);

    ssm_faces = ssm.faces;
    ssm_faces(cellfun('isempty', ssm_faces)) = [];

    ssm_partial = {};
    ssm_partial.vertices = [];
    ssm_partial.faces = {};
    ssm_partial.numverts = {};

    test_shape = {};
    test_shape.vertices = [];
    test_shape.faces = {};
    test_shape.numverts = {};

    W = []; % weights
    BTX = []; BTY = []; BTZ = []; % pca vector columns x y z

    % construct the partial multi-shapes
    for i = 1:length(structure_list_partial)

        % find index in full_structure_list
        j = find(strcmp(structure_list_full, structure_list_partial(i)));

        % Load test shape
        target = testing_data.(structure_list_full{j}).(strcat('oai_', target_subject_tag));

        test_shape.vertices = [test_shape.vertices; target.VERT];
        test_shape.faces{i} = target.TRIV;
        test_shape.numverts{i} = size(target.VERT, 1);

        ssm_partial.vertices = [ssm_partial.vertices; ssm.mean_shape(begin_ptr(j):end_ptr(j), :)];
        ssm_partial.faces{i} = ssm_faces{j};
        ssm_partial.numverts{i} = size(ssm.mean_shape(begin_ptr(j):end_ptr(j), :), 1);

        W = [W; ssm.weights(begin_ptr(j):end_ptr(j))];

        [p, ~] = size(ssm.eig_vectors_scaled);
        s = p / 3;

        BTXX = ssm.eig_vectors_scaled(1:s, :);
        BTXY = ssm.eig_vectors_scaled(s + 1:2 * s, :);
        BTXZ = ssm.eig_vectors_scaled(2 * s + 1:end, :);

        BTX = [BTX; BTXX(begin_ptr(j):end_ptr(j), :)];
        BTY = [BTY; BTXY(begin_ptr(j):end_ptr(j), :)];
        BTZ = [BTZ; BTXZ(begin_ptr(j):end_ptr(j), :)];

    end

    % W contains the correlation coefficient of the structures
    W = spdiags(sqrt([W; W; W]), 1, size(W, 1) * 3, size(W, 1) * 3);

    eig_vectors_scaled = [BTX; BTY; BTZ];
    ssm_partial.eig_vectors_scaled = W * eig_vectors_scaled;
  end


