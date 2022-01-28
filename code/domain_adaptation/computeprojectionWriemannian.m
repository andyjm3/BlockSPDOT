function X_src_r = computeprojectionWriemannian(X_tgt, gamma_r)
    m = size(gamma_r, 1);
    n = size(gamma_r, 2);
    d = size(X_tgt, 1);
    X_src_r = nan(d, d, m);
    for ii = 1 : m
 		temp_w = WeightedRiemannianMean(X_tgt, gamma_r(ii,:));
        X_src_r(:,:,ii) = temp_w./sum(gamma_r(ii,:)); % 
    end
end
