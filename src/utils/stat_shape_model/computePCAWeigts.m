function [w] = computePCAWeigts(data, root_structure)
    if root_structure ~= 0
       
        numverts = {data.numverts};
        numverts(cellfun('isempty', numverts)) = [];

        [begin_ptr, end_ptr] = computeRegionPointers(numverts);

        n = 1;
        X = {};
        while n <= numel(numverts)
            % vectorize input to pca
            x = zeros(data(n).numverts * 3, length(data));
            for i = 1: length(data)
                x(:, i) = [data(i).vertices_aligned(begin_ptr(n):end_ptr(n), 1); ...
                        data(i).vertices_aligned(begin_ptr(n):end_ptr(n), 2); ...
                        data(i).vertices_aligned(begin_ptr(n):end_ptr(n), 3)];
            end
            [Evalues, Evectors, ~, ~, ~] = PCA(x);
            i = find(cumsum(Evalues) > sum(Evalues) * 0.95, 1, 'first');
            Evectors = Evectors(:, 1:i); Evalues = Evalues(1:i);
            X{n} = Evectors' * x;
            n = n + 1;
        end

        ro = zeros(length(numverts));
        for i = 1:length(numverts)
            for j = 1:length(numverts)
                [~, ~, r] = canoncorr(X{i}', X{j}');
                ro(i, j) = mean(r);

                % if options.verbose == true
                %     list2 = {'FB', 'TB', 'FC', 'TCL', 'TCM'};
                %     imagesc(ro)
                %     set(gca, 'xtick', [1:length(numverts)], 'xticklabel', list2, ...
                %         'xTickLabelRotation', 45)
                %     set(gca, 'ytick', [1:length(numverts)], 'yticklabel', list2, ...
                %         'yTickLabelRotation', 45)
                %     colorbar
                % end
            end

        end

        w = zeros(size(data(1).vertices, 1), 1);
        for i = 1:numel(numverts)
            w(begin_ptr(i):end_ptr(i)) = ro(root_structure, i);
        end

        w = sqrt([w; w; w]);


    else % simple pca
        w = ones(size(data(1).vertices, 1), 1);
        w = sqrt([w; w; w]);
    end

end

