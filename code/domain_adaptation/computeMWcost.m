function obj_mw_mat = computeMWcost(C, Gamma)
    m = size(Gamma, 3);
    n = size(Gamma, 4);
    obj_mw_mat = nan(m,n);
    for ii = 1 : m
        for jj = 1 : n
            obj_mw_mat(ii, jj) = trace((Gamma(:,:,ii,jj)) * C(:,:,ii,jj));
        end
    end
end
 