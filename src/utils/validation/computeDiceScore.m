function  [dsc] = computeDiceScore(A, B)
    intersect = (A & B);
    union = (A | B);
    d = sum(union(:));
    a = sum(intersect(:));
    b = sum(A(:));
    c = sum(B(:));
    dsc = 2 * a / (b + c) * 100;
end