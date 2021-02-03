function showshapes(varargin)
n=1;
color = {};
while n<= numel(varargin)
   color{n} = ones(size(varargin{n}.VERT,1),1)+n;

    showshape(varargin{n}, color{n});
    hold on
    n=n+1;
end
lighting gouraud
camlight headlight