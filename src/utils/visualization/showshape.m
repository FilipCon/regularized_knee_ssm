function showshape(shape,color,viewpoint)
   
    if isfield(shape,'VERT')
        X = [shape.VERT(:,1) shape.VERT(:,2) shape.VERT(:,3)];
    elseif isfield(shape,'vertices')
        X = [shape.vertices(:,1) shape.vertices(:,2) shape.vertices(:,3)];
    elseif isfield(shape,'X') && isfield(shape,'Y') && isfield(shape,'Z')
        X = [shape.X(:) shape.Y(:) shape.Z(:)];                   
    end
    
    if isfield(shape,'TRIV')
        F = shape.TRIV;
    elseif isfield(shape,'faces')
        F = shape.faces;
    end
    
    nv = size(X,1);                                             % # of vertices
    nf = size(F,1);                                             % # of faces
    
    if (nargin==1)
        color=zeros(nv,1);
    end
    
%      viewpoint = [180 -90];
%     figure;
    trisurf(F, X(:,1), X(:,2), X(:,3), color); 
%      scatter3(X(:,1), X(:,2), X(:,3),ones(1,size(X,1)),color); 
    if exist('viewpoint','var')
       view(viewpoint); 
    end
%      view(viewpoint);
    axis image; 
    axis off;
    shading flat;

rotate3d on

end
