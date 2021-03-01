%% PARAMETERS.m
%
% Settings file of the Descriptor used at hand.
%
%% Author: Filip Konstantinos <filip.k@ece.upatras.gr>
%% Last updated: 2021-02-03

switch DescriptorType

    case 'HKS'
        t0 = 0.01;
        alpha1 = 2;
        tauScale = 75;
        tau = 0:1/2:tauScale;
		k =100;

    case 'SIHKS'
        t0 = 0.01;
        TimeScale = 15;
        alpha1 = 2;
        tau = 0:(1/16):TimeScale;
        numF = 50;
		k = 100;

    case 'WKS'
        N = 100;
        wks_variance = N * 0.05;
		k = 100;

    case 'SGWS'
		k = 100;
        Nscales = 10;
        designtype = 'abspline3';
        %esigntype='mexican_hat';
        %designtype='meyer';
        %designtype='simple_tf';

    case 'SHOT'
        n_bins = 10;
        radius = 15;
        min_neighs = 10;
end
