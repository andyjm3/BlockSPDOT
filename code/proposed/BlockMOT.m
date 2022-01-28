function [Xopt, costopt, Lambdaopt, Thetaopt, Psiopt, infos] = BlockMOT(C,options)
% computes the optimal transport distance between sets of objects
% C is the block cost matrix d-by-d-by-m-by-n
    [d,~,m,n] = size(C);
    
    myI = eye(d);
    lambda = 1e-3;
    symm = @(X) .5*(X+X');
	
    for i = 1:m
        M(:,:,i) = myI/m;
    end
    
    for j = 1:n
        N(:,:,j) = myI/n;
    end
    
    localdefaults.lambda = 0;
    localdefaults.M = M;
    localdefaults.N = N;
    localdefaults.tolblockDS = 1e-9;
    localdefaults.tolgradnorm = 1e-6;
    localdefaults.maxiter = 30;
    localdefaults.tolpcg = 1e-6;
    localdefaults.maxiterspcg = 500;
    localdefaults.init = [];
    localdefaults.method = 'CG';

    options = mergeOptions(localdefaults, options);

    if size(M, 3) ~= m || size(M, 1) ~= d
        error('M should be a multidimensional array of size %d-by-%d-by-%d.', d, d, m);
    end
    if size(N, 3) ~= n || size(N, 1) ~= d
        error('N should be a multidimensional array of size %d-by-%d-by-%d.', d, d, n);
    end
    
    M = options.M;
    N = options.N;
    lambda = options.lambda;
    tolblockDS = options.tolblockDS;
    tolgradnorm = options.tolgradnorm;
    maxiter = options.maxiter;
    tolpcg = options.tolpcg;
    maxiterspcg = options.maxiterspcg;
    
    
    problem.M = blockdoublystochasticwithSPDblocksfactory(d, m, n, M, N, tolblockDS, tolpcg, maxiterspcg);
    problem.cost = @cost;
    problem.egrad = @egrad;
    problem.ehess = @ehess;

    myeps = 1e-8;
    
    function [f] = cost(X)
        f = 0;
        
        %if ~isfield(store, 'logGamma')
        %    store.logGamma =  cell(m,n); % for storing logm
        %end
        
        for mm = 1:m
            for nn = 1:n
                Cij = C(:,:,mm,nn);
                Xij = X(:,:,mm,nn);
                if lambda < 1e-10
                    f = f + Cij(:)' * Xij(:);
                else
                    %if isempty(store.logGamma{mm, nn})
                    %    store.logGamma{mm, nn} = logm(Xij);
                    %end
                    %logmXij = store.logGamma{mm, nn};
                    logmXij = logM(Xij + myeps*eye(d));
                    f = f + Cij(:)' * Xij(:) + lambda*trace(Xij*logmXij - Xij); % added minus tr(Xij)
                end
            end
        end
    end

    function [grad] = egrad(X)
        grad = zeros(size(X));
        
        %if ~isfield(store, 'logGamma')
        %    store.logGamma =  cell(m,n); % for storing logm
        %end
        
        for mm = 1:m
            for nn = 1:n
                Cij = C(:,:,mm,nn);
                Xij = X(:,:,mm,nn);
                if lambda < 1e-10
                    grad(:,:,mm,nn) = Cij;
                else
                    %if isempty(store.logGamma{mm, nn})
                    %    store.logGamma{mm, nn} = logm(Xij);
                    %end
                    %logmXij = store.logGamma{mm, nn};
                    logmXij = logM(Xij + myeps*eye(d));
                    grad(:,:,mm,nn) = Cij + lambda*(logmXij);
                end
            end
        end
    end

    function [hess] = ehess(X, eta)
        hess = nan(size(X));
        
        for mm = 1:m
            for nn = 1:n
            	Xij = X(:,:,mm,nn);
            	etaij = eta(:,:,mm,nn);
                if lambda < 1e-10
                    error('Hessian is ill-defined for linear problems');
                else
                    gdotij = lambda*symm(dlogm(Xij, etaij));
                end
                hess(:,:,mm,nn) = gdotij;
            end
        end
    end
    %checkgradient(problem);
    %keyboard;
    %checkhessian(problem);
    %keyboard;

    options.maxiter = maxiter;
    options.tolgradnorm = tolgradnorm;
    options.linesearch = @linesearch;

    if strcmpi(options.method, 'SD') 
        [Xopt, costopt, infos] = steepestdescent(problem,options.init,options);
    elseif strcmpi(options.method, 'CG') 
    	[Xopt, costopt, infos] = conjugategradient(problem,options.init,options);
	end

    egradopt = egrad(Xopt);
    egradoptscaled = nan(size(egradopt));
    for mm = 1 : m
    	for nn = 1 : n
    		egradoptscaled(:,:,mm, nn) = Xopt(:,:,mm, nn)*egradopt(:,:,mm, nn)*Xopt(:,:,mm, nn);
    	end
    end 
    [~, Lambdaopt, Thetaopt] = problem.M.orthproj(Xopt, egradoptscaled);
    Lambdaopt = -Lambdaopt; %
    Thetaopt = -Thetaopt;

    Psiopt = nan(size(Xopt));
    for mm = 1 : m
        for nn = 1 : n
            Psiopt(:,:,mm, nn) = (egradopt(:,:,mm, nn) - Lambdaopt(:,:,mm) -Thetaopt(:,:,nn)) ;
        end
    end 
    % keyboard;
end

