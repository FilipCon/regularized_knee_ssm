function  [evecs, evecs_trans, evals] = getEigenFunctions( shape, k, evecs_file)

    if ~exist('evecs_file', 'var')
        [evecs, evecs_trans, evals] = calc_eigenfun(shape, k);
    else
        stored_evecs = load(evecs_file);
        evecs = stored_evecs.evecs(:, 1:k);
        evecs_trans = stored_evecs.evecs_trans(1:k, :);
        evals = stored_evecs.evals(1:k, 1:k);
    end

end