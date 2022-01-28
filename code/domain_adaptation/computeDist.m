function myDist = computeDist(X, Y)
    m = size(X, 3);
    n = size(Y, 3);
    for ii = 1 : m
        for jj = 1 : n
            myDist(ii, jj) = norm(X(:,:,ii) - Y(:,:,jj), 'fro')^2 ;
        end
    end
end