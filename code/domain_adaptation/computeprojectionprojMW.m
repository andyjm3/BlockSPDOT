function [X_src_projmw, X_tgt_projmw] = computeprojectionprojMW(X_tgt, W, Gamma)
    m = size(Gamma, 3);
    n = size(Gamma, 4);
    d = size(X_tgt, 1);
    k = size(X_tgt, 2);
    r = size(W, 2);
    
    X_src_projmw = nan(r, k, m);
    X_tgt_projmw = nan(r, k, n);

    for jj = 1 : n
        X_tgt_projmw(:,:,jj) = W'*X_tgt(:,:,jj);
    end

    for ii = 1 : m
        temp_mw = 0;
        for jj = 1 : n
            temp_mw = temp_mw + Gamma(:,:,ii,jj)*X_tgt_projmw(:,:,jj);
        end
        X_src_projmw(:,:,ii) = sum(Gamma(:,:,ii,:), 4) \ temp_mw; % This is not spd but r by k matrix.
    end
end
