function [Xbar, Gamma, info] = BlockWBarycenter(X, C, w, options, Xbarinit)
    % == Input ==
    % X is a cell array of K SPD measures, where each entry is d-by-d-by-n
    % (assumed uniform sampling)
    % C is cost matrix of size d-by-d-by-n-by-n (assumed uniform for all)
    % w is the weight of length K
    
    % == Output ==
    % Xbar: barycenter of size d-by-d-by-n (assumed fixed)
    % Gamma: Coupling
    % info: Information for optimization
    
    [d, ~, n] = size(X{1});
    K = length(X);
    assert(K == length(w), "Size of barycenter should match the size of weight!");
    assert(sum(w) == 1, "Weight should sum to 1!");
    
    %symm = @(Delta) 0.5*(Delta + Delta');

    % Set local defaults.
    localdefaults.lambda = 1e-5;
    localdefaults.tolblockDS = 1e-9;
    localdefaults.MOTtolgradnorm = 1e-6;
    localdefaults.MOTiter = 20;
    localdefaults.tolpcg = 1e-6;
    localdefaults.WBiter = 20;
    localdefaults.WBtol = 1e-5; % tol for gradient norm for WB opt
    localdefaults.verbosity = 0;
    localdefaults.WBmethod = 'CG';
    localdefaults.MOTmethod = 'CG';

    options = mergeOptions(localdefaults, options);
    
    lambda = options.lambda;
    %myeps = 1e-8;
    myeps1 = 0;

    if  ~exist('Xbarinit', 'var') ||  isempty(Xbarinit) 
        Xbarinit = [];
    end
    
    % Update Xbar by Riemannian steepest descent
    problem.M = sympositivedefinitesimplexfactory(d, n); % can only deal with sum is identity
    problem.cost = @cost;
    problem.egrad = @egrad;
    
    
    function [f, store] = cost(Xbar, store)
        if ~isfield(store, 'Lambda_opt')
        
            obj = 0;
            Gamma = zeros(d,d,n,n,K);
            Lambda_opt = 0;
            for ell = 1:K

                % only compute when weight is positive
                if w(ell) > 0

                    % Compute coupling Gamma_ell using BlockMOT
                    MOTopt.lambda = lambda;
                    MOTopt.maxiter = options.MOTiter;
                    MOTopt.M = Xbar;
                    MOTopt.N = X{ell};
                    MOTopt.tolblockDS = options.tolblockDS;
                    MOTopt.tolpcg = options.tolpcg;
                    MOTopt.tolgradnorm = options.MOTtolgradnorm;
                    MOTopt.verbosity = options.verbosity;
                    MOTopt.method = options.MOTmethod;
                    [Gamma_ell, obj_ell, Lambda_ell] = BlockMOT(C, MOTopt);


                    Gamma(:,:,:,:,ell) = Gamma_ell;

                    % Compute objective
                    obj = obj + w(ell) * obj_ell;

                    % Update Lambda_opt, the Lagrange dual to X_bar. 
                    Lambda_opt = Lambda_opt + w(ell) * Lambda_ell;
                end
                obj = obj + myeps1*0.5*norm(Xbar(:),'fro')^2;
            end
            store.Gamma = Gamma;
            store.Lambda_opt = Lambda_opt;
            store.obj = obj;  
        end
        f = store.obj;
    end

    
    function [g, store] = egrad(Xbar, store)
        if ~isfield(store, 'Lambda_opt')
            [~, store] = cost(Xbar, store);
        end        
        g = store.Lambda_opt + myeps1*Xbar;
    end 

    % checkgradient(problem);
    % % saveas(gcf, 'gradientcheckBlockWBarycenter.pdf');
    % pause;

    mfdopt.maxiter = options.WBiter;
    mfdopt.tolgradnorm = options.WBtol;
    mfdopt.linesearch = @linesearch;

    if strcmpi(options.WBmethod, 'SD')    
    	[Xbar, ~, info] = steepestdescent(problem, Xbarinit, mfdopt);
	elseif strcmpi(options.WBmethod, 'CG' )
		[Xbar, ~, info] = conjugategradient(problem, Xbarinit, mfdopt);
	end

    store = [];
    [~, store] = cost(Xbar, store);
    Gamma = store.Gamma;
end