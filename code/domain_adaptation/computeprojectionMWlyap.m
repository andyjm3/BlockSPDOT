function X_src_mw = computeprojectionMWlyap(X_tgt, Gamma)
    m = size(Gamma, 3);
    n = size(Gamma, 4);
    d = size(X_tgt, 1);
    X_src_mw = nan(d, d, m);

    for ii = 1 : m
        temp_mw = 0;
        for jj = 1 : n
            temp_mw = temp_mw + Gamma(:,:,ii,jj)*X_tgt(:,:,jj);
        end
        X_src_mw(:,:,ii) = lyap(sum(Gamma(:,:,ii,:), 4), -(temp_mw + temp_mw')); % This is spd of size d by d.
    end
end

