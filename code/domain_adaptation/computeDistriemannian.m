function myDist = computeDistriemannian(X, Y)
    m = size(X, 3);
    n = size(Y, 3);
    for ii = 1 : m
        for jj = 1 : n
            myDist(ii, jj) = RiemannianDist(X(:,:,ii), Y(:,:,jj), 2)^2;
        end
    end
end