function desc = getDescriptor(X, DescriptorType, desc_file)

if ~exist('desc_file', 'var')
    PARAMETERS;
    
    if DescriptorType == 'SHOT'
        
        desc = calc_shot(X.VERT', X.TRIV', ...
            [1:size(X.VERT, 1)]', n_bins, ...
            radius, min_neighs)';
    else
        [evecs, ~, evals] = calc_eigenfun(shape, k);
        PHI = evecs; % eigenfunctions
        E = evals; % eigenvalues
        
        [num_vertices,n_eigenvalues] = size(PHI);
        
        % compute the spectral descriptors
        switch DescriptorType
            
            case 'GPS'
                desc =  PHI(:,start_eval:end_eval)./repmat((...
                    sqrt(E(:,start_eval:end_eval))+eps),num_vertices,1);
                
            case 'HKS'
                ee = t0*alpha1.^tau;
                NumTau = length(tau);
                
                HKS=zeros(num_vertices,NumTau);
                
                for ii = 1:NumTau
                    HKS(:,ii) = sum(PHI.^2.*...
                        repmat( exp(-(ee(ii).*E')),num_vertices,1),2);
                end
                desc =  HKS;
                
            case 'SIHKS'
                HKS=zeros(num_vertices,TimeScale);
                t = t0*alpha1.^tau;
                for ii = 1:length(tau)
                    HKS(:,ii) = sum( -log(alpha1).*PHI.^2.*repmat(...
                        (t(ii).*E').*exp(-(t(ii).*E')),num_vertices,1),2) ...
                        ./sum(PHI.^2.*repmat( exp(-(t(ii).*E')),num_vertices,1),2);
                end
                
                SHKS = zeros(num_vertices,(length(tau)-1));
                for ii = 1:(length(tau)-1)
                    SHKS(:,ii) = HKS(:,(ii+1))-HKS(:,(ii));
                end
                
                SIHKS = zeros(num_vertices,(length(tau)-1));
                for ii = 1:num_vertices
                    SIHKS(ii,:) = abs(fft(SHKS(ii,:)));
                end
                desc =  SIHKS(:,1:numF);
                
            case 'WKS'
                WKS=zeros(num_vertices,N);
                
                log_E=log(max(abs(E),1e-6))';
                e=linspace(log_E(2),(max(log_E))/1.02,N);
                sigma=(e(2)-e(1))*wks_variance;
                
                C = zeros(1,N); %weights used for the normalization of f_E
                
                for ii = 1:N
                    WKS(:,ii) = sum(PHI.^2.*...
                        repmat( exp((-(e(ii) - log_E).^2) ./ (2*sigma.^2)),num_vertices,1),2);
                    C(ii) = sum(exp((-(e(ii)-log_E).^2)/(2*sigma.^2)));
                end
                desc =  WKS;
                
            case 'SGWS'
                MWaveletS = [];
                for nscales = 1:Nscales
                    lmax =  E(n_eigenvalues);
                    [g,t]=sgwt_filter_design(lmax,nscales,'designtype',designtype);
                    arange=[0 lmax];
                    J=numel(g);
                    
                    WaveletS = zeros(num_vertices,J);
                    for j=1:J
                        for ii = 1:n_eigenvalues
                            WaveletS(:,j) = WaveletS(:,j)+...
                                PHI(:,ii).^2.*repmat( g{j}(E(ii)),num_vertices,1);
                        end
                    end
                    MWaveletS = [MWaveletS,WaveletS];
                end
                desc =  MWaveletS;
        end
    end
else
    desc_mat = load(desc_file);
    desc = desc_mat.desc;
end

end



