function obj_w_mat = computeWcost(C, gamma_w)
    m = size(gamma_w, 1);
    n = size(gamma_w, 2);
    obj_w_mat = nan(m,n);
    for ii = 1 : m
        for jj = 1 : n
            obj_w_mat(ii, jj) = gamma_w(ii,jj) * C(ii,jj);
        end
    end
end