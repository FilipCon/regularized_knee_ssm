function [ keval, x, dd, rgeod ] = kimeval_final(shapet, Lgr, L, opt )
% Kim's evaluation
% assumed that the correspondence from a shape source to a shape target
if ~exist( 'opt', 'var' )
   opt.dstep = 0.005; 
end

%-
if ~isfield( opt, 'dstep' )
    dstep = 0.005;
else
    dstep = opt.dstep; % step for checking how many points are within the given geod error radius
end
%-
if ~isfield( opt, 'rgeod' )

    Area= calcArea( shapet );
    Area=sqrt(sum(Area));
else
    rgeod = opt.rgeod; % geaod radius of the target shape
end

if ~isfield( opt, 'maxpercentage' )
    maxpercentage = 0.25;
else
    maxpercentage = opt.maxpercentage; % maximal percentage of the error to consider
end

errors_ind = 0:dstep:maxpercentage;
errors_ind = errors_ind(:);
errors_ind(end+1)=errors_ind(end)+dstep;

N = size( L, 1 ); % number of points which were matched

tic
dd=zeros(N,1);

if ~isfield( opt, 'dist_maps' )
    for i = 1:N
        disp('Computing distance maps...')
        inds = L( i, 1 ); %source
        indt = L( i, 2 ); %mapped point
        indtorig = Lgr( inds, 2 );%ground-truth correspondence
        %
        d = d_shape( shapet, indt ); % distance from the mapped point to all other
        %
        d_dist=d(indtorig)/Area;%distortion=ditance to the ground-truth
        %
        dd(i)=d_dist;
    end
else
    D = opt.dist_maps;
    inds = L( :, 1 ); %source
    indt = L( :, 2 ); %mapped point
    indtorig = Lgr( inds, 2 );%ground-truth correspondence
    
    %     d = D(:,indt);
    for i = 1:N
        dd(i)=D(indtorig(i),indt(i))/Area;
    end
end
N=length(dd);
x = errors_ind;

stam=hist(dd,x);
keval=(cumsum(stam)/N)*1;

x=x(1:end-1);
keval=keval(1:end-1);
toc
end

