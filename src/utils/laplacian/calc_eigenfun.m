function [Phi, Phi_t, Lambda, M, S] = calc_eigen_functions(shape,k)
    % Linear FEM
    [M,S]=laplacian([shape.VERT(:,1) shape.VERT(:,2) shape.VERT(:,3)],shape.TRIV);

    M = diag(sum(M,2));
    [Phi,Lambda] = eigs(-S,M,k,1e-5);
    Lambda = diag(Lambda);
    [Lambda,idx] = sort(Lambda,'descend');
    Phi = Phi(:,idx);
    Phi_t = Phi' * M;
    Lambda = diag(abs(Lambda));

    end
