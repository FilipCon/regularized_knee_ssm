function  [voe] = computeVolumeOverlap(A, B)
    intersect = (A & B);
    union = (A | B);
    a = sum(intersect(:));
    b = sum(union(:));
    voe = 100 * (1 - a / b);
end
