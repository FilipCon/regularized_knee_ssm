function [begin_ptr, end_ptr] = computeRegionPointers(numverts)
    % remove empty fields in cell
    numverts(cellfun('isempty', numverts)) = [];

    begin_ptr(1) = 1;
    end_ptr(1) = numverts{1};
    for n = 2:numel(numverts)
        begin_ptr(n) = end_ptr(n - 1) + 1;
        end_ptr(n) = numverts{n} + end_ptr(n - 1);
    end

end