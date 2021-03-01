function x_vect = vectorize_shape(x)
    x_vect = reshape(x, size(x,1) * 3,1);
end