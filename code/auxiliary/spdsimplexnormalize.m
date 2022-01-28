function Y = spdsimplexnormalize(X, s)
% normalize a set of SPD matrices to simplex with sum equal identity
% X is a d-by-d-by-N array
% s is scale

if ~exist('s', 'var') || isempty(s)
    s = 1;
end

Y = nan(size(X));
d = size(X,1);
N = size(X, 3);
sumX = sum(X,3);
sumXsqrtm = sqrtm(sumX);
        
for i = 1:N
    Y(:,:,i) = sumXsqrtm\ (X(:,:,i) / sumXsqrtm);
    Y(:,:,i) = s * 0.5*(Y(:,:,i) + (Y(:,:,i))' );
end

assert(norm(sum(Y,3) - s * eye(d),'fro') < 1e-10);

end

