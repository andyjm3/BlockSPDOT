function [Gammaopt, costopt] = compute_gw_coupling(D1, D2, d, M, N, options)
    % compute Gromov-Wasserstein coupling for l2 norm as loss
    
    tolblockDS = options.tolblockDS;
    tolgradnorm = options.tolgradnorm;
    maxiter = options.maxiter;
    tolpcg = options.tolpcg;
    lambda = options.lambda;
    
    m = size(D1,1);
    n = size(D2,1);

    problem.M = blockdoublystochasticwithSPDblocksfactory(d, m, n, M, N, tolblockDS, tolpcg);
    problem.cost = @cost;
    problem.egrad = @egrad;
    
    
    % make block distance matrix offline
    D1block = zeros(d,d,m,m);
    D2block = zeros(d,d,n,n);
    for i = 1 : m
        for j = 1 : m
            D1block(:,:,i,j) = D1(i,j) * eye(d);
        end
    end
    for i = 1 : n
        for j = 1 :n
            D2block(:,:,i,j) = D2(i,j) * eye(d);
        end
    end
    D1blockmat = blockresh(D1block);
    D2blockmat = blockresh(D2block); 
    


    function [f, store] = cost(Gamma, store)
        if ~isfield(store, 'logGamma')
            if lambda >= 1e-10
                store.entropyReg =  nan(m,n); 
                store.logGamma = nan(size(Gamma));
                for ii = 1 : m
                    for jj = 1 : n
                        loggamma = logM(Gamma(:,:,ii,jj));
                        store.logGamma(:,:,ii,jj) = loggamma; 
                        store.entropyReg(ii,jj) = trace(Gamma(:,:,ii,jj) * loggamma - Gamma(:,:,ii,jj));
                    end
                end
            end
        end
        

        D1blockmatsq = D1blockmat .^2;
        temp = reshape(permute(sum(Gamma,4), [1 3 2]), [d*m, d]);
        term1 = D1blockmatsq * temp; % (d*m,d)

        D2blockmatsq = D2blockmat .^2;
        temp = reshape(permute(reshape(sum(Gamma,3),[d d n]), [1 3 2]), [d*n, d]);
        term2 = D2blockmatsq * temp; % (d*n,d)

        D1blockmath1 = D1blockmat;
        D2blockmath2 = 2 * D2blockmat;
        Gammablock = blockresh(Gamma);
        term3 = D1blockmath1 * Gammablock * D2blockmath2; % (d*m, d*n)


        term1rep = repmat(term1, [1 n]);
        term1repblock = blockreshT(term1rep, d, m, n);
        term2rep = repmat(term2, [1 m]);
        term2repblock = permute(blockreshT(term2rep, d, n, m), [1 2 4 3]);
        term3resh = blockreshT(term3, d, m, n);
        termsum = term1repblock + term2repblock - term3resh;
        store.termsum = termsum; % store for gradient compute
        termfinal = multiprod(Gamma, termsum);
        if lambda < 1e-10
            f = sum(multitrace(reshape(termfinal, [d d m*n])));
        else
            f = sum(multitrace(reshape(termfinal, [d d m*n]))) + sum(lambda * store.entropyReg, 'all');
        end
        
    end

    
    function [grad, store] = egrad(Gamma, store)
        if ~isfield(store, 'termsum')
            [~, store] = cost(Gamma, store);
        end
        if ~isfield(store, 'logGamma')
            if lambda >= 1e-10
                store.entropyReg =  nan(m,n); 
                store.logGamma = nan(size(Gamma));
                for ii = 1 : m
                    for jj = 1 : n
                        loggamma = logM(Gamma(:,:,ii,jj));
                        store.logGamma(:,:,ii,jj) = loggamma; 
                        store.entropyReg(ii,jj) = trace(Gamma(:,:,ii,jj) * loggamma - Gamma(:,:,ii,jj));
                    end
                end
            end
        end
                
        if lambda < 1e-10
            grad = store.termsum;
        else
            grad = store.termsum + lambda * store.logGamma;
        end
        
        
    end

    %checkgradient(problem);
    %keyboard;
    
    mfdoptions.maxiter = maxiter;
    mfdoptions.tolgradnorm = tolgradnorm;
    
    if strcmpi(options.method, 'SD') 
        [Gammaopt, costopt] = steepestdescent(problem,options.init,mfdoptions);
    elseif strcmpi(options.method, 'CG') 
    	[Gammaopt, costopt] = conjugategradient(problem,options.init,mfdoptions);
    end
    
    
    
    function Y = blockresh(X)
        % reshape a 4d array into 2d with block matrix format
        Y = reshape(permute(X, [1 3 2 4]), [size(X,1)*size(X,3) size(X,2)*size(X,4)]);
    end
    function X = blockreshT(Y, d, m, n)
        % reshape into (d,d,m,n) from a block matrix form (inverse of
        % blockresh function)
        assert(size(Y,1) == d*m);
        assert(size(Y,2) == d *n);
        X = permute(reshape(Y, [d m d n]), [1 3 2 4]);
    end

end

