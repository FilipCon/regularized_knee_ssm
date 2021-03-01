function [d] = d_shape(shape, src_idx)
if ismac
    d=d_shape2(shape,src_idx);
else
    num_vert = length(shape.VERT);
    src = repmat(Inf, num_vert, 1);
    src(src_idx) = 0;
    d = fastmarch(shape.TRIV, shape.VERT(:,1), shape.VERT(:,2), shape.VERT(:,3), double(src), set_options('mode', 'single'));
%     d = fastmarchmex('init', int32(TRIV-1), double(X(:)), double(Y(:)), double(Z(:)));
%     f = fastmarchmex('march',f,double(u));
end % function [d] = d_shape(shape, src_id
end


function [d] = d_shape2(shape, src_idx)

if ismac
    num_vert = length(shape.X);
    src = repmat(Inf, num_vert, 1);
    src(src_idx) = 0;
    
    %-
    f = fastmarchmex('init', int32(shape.TRIV-1), double(shape.VERT(:,1)), double(shape.VERT(:,2)), double(shape.VERT(:,3)));
    
    d = fastmarchmex('march', f, double(src));
    d(d>=9999999) = Inf;
    
    % d = fastmarch(shape.TRIV, shape.X, shape.Y, shape.Z, double(src), set_options('mode', 'single'));
    % d = fastmarchmex('init', int32(TRIV-1), double(X(:)), double(Y(:)), double(Z(:)));
    % f = fastmarchmex('march',f,double(u));
    fastmarchmex('deinit', f);
else
    num_vert = length(shape.X);
    src = repmat(Inf, num_vert, 1);
    src(src_idx) = 0;
    d = fastmarch(shape.TRIV, shape.VERT(:,1), shape.VERT(:,2), shape.VERT(:,3), double(src), set_options('mode', 'single'));
    % d = fastmarchmex('init', int32(TRIV-1), double(X(:)), double(Y(:)), double(Z(:)));
    % f = fastmarchmex('march',f,double(u));
end % function [d] = d_shape(shape, src_id
end