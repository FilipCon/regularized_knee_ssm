function x = unvectorize_shape(x_vec)
    x = reshape(x_vec, size(x_vec, 1) / 3, 3);
end