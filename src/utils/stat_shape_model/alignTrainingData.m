function [result] = alignTrainingData(data)
    iter = 0;
    err0 = inf;
    tol = 1e-10;
    max_iter = 100;
    MeanVertices = data(1).vertices;

    result = data;
    while true

        for i = 1:length(result)
            [~, result(i).vertices_aligned] = procrustes(MeanVertices, ...
                result(i).vertices, 'scaling', 0);
        end

        x = zeros(size(result(1).vertices, 1) * 3, length(result));

        for i = 1:length(result)
            x(:, i) = [result(i).vertices_aligned(:, 1); ...
                    result(i).vertices_aligned(:, 2); ...
                    result(i).vertices_aligned(:, 3)];
        end

        x_mean = sum(x, 2) / size(x, 2);

        err = vecnorm(x_mean - reshape(MeanVertices, size(MeanVertices, 1) * 3, 1));

        if (abs(err - err0) < tol || iter > max_iter)
            break
        end

        err0 = err;
        MeanVertices = reshape(x_mean, size(x_mean, 1) / 3, 3);
        iter = iter + 1;
    end

end
