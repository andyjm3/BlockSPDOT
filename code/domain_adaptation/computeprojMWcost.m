function obj_projmw_mat = computeprojMWcost(C, W, Gamma)
    m = size(Gamma, 3);
    n = size(Gamma, 4);
    obj_projmw_mat = nan(m,n);
    for ii = 1 : m
        for jj = 1 : n
            obj_projmw_mat(ii, jj) = trace((Gamma(:,:,ii,jj)) * (W'*(C(:,:,ii,jj)*W)));
        end
    end
end
