function X_src_mw = computeprojectionMW(X_tgt, Gamma)
    m = size(Gamma, 3);
    n = size(Gamma, 4);
    d = size(X_tgt, 1);
    k = size(X_tgt, 2);
    X_src_mw = nan(d, k, m);

    for ii = 1 : m
        temp_mw = 0;
        for jj = 1 : n
            temp_mw = temp_mw + Gamma(:,:,ii,jj)*X_tgt(:,:,jj);
        end
        X_src_mw(:,:,ii) = sum(Gamma(:,:,ii,:), 4) \ temp_mw; % This is d by k;
    end
end
