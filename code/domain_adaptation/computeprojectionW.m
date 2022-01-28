function X_src_w = computeprojectionW(X_tgt, gamma_w)
    m = size(gamma_w, 1);
    n = size(gamma_w, 2);
    d = size(X_tgt, 1);
    k = size(X_tgt, 2);
    X_src_w = nan(d, k, m);
    for ii = 1 : m
        temp_w = 0;
        for jj = 1 : n
            temp_w = temp_w + gamma_w(ii,jj)*X_tgt(:,:,jj);
        end
        X_src_w(:,:,ii) = temp_w./sum(gamma_w(ii,:)); % 
    end
end
