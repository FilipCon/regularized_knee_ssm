function  [vd] = computeVolumeDifference(A, B)
    a = sum(A(:));
    b = sum(B(:));
    vd = 100 * (b - a) / a;
end