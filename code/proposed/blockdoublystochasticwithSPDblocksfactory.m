function Mfd = blockdoublystochasticwithSPDblocksfactory(d, m, n, M, N, tolblockDS, tolpcg, maxiterspcg)
% Manifold of m-by-n blocks of SPD matrices of size d-by-d where sum of row
% i blocks equals Mi and sum of column j blocks equals Nj.
%
% A point on the manifold is represented as a 4-dimensional array with size
% d-by-d-by-m-by-n and each d-by-d matrix is symmetric positive definite (SPD).
%
% M and N are given arrays for constraints and are represented as
% d-by-d-by-m and d-by-d-by-n respectively. It is required that sum(Mi) = 
% sum(Nj). If either one of inputs is not given, defaults are used, i.e.,
% Mi = eye(d)/m, Nj = eye(d)/n.

	myI = eye(d);
	
	if ~exist('M', 'var') || isempty(M)
        for i = 1:m
            M(:,:,i) = myI/m;
        end
    end

    if ~exist('N', 'var') || isempty(N)
        for j = 1:n
            N(:,:,j) = myI/n;
        end
    end

    if size(M, 3) ~= m || size(M, 1) ~= d
        error('M should be a multidimensional array of size %d-by-%d-by-%d.', d, d, m);
    end
    if size(N, 3) ~= n || size(N, 1) ~= d
        error('N should be a multidimensional array of size %d-by-%d-by-%d.', d, d, n);
    end

    if ~exist('tolblockDS', 'var') || isempty(tolblockDS)
        tolblockDS = 1e-10;
    end 
    
    if ~exist('tolpcg', 'var') || isempty(tolpcg)
        tolpcg = 1e-8;
    end 

    if ~exist('maxiterspcg', 'var') || isempty(maxiterspcg)
        maxiterspcg = 500;
    end 

    maxitersblockDS = 1000;
    checkperiodblockDS = 1;
    myeps = 1e-8;


    symm = @(X) .5*(X+X');
    
    Mfd.name = @() sprintf('%dx%d block matrix with block doubly stochastic constraints, each block is a symmetric positive definite matrix of size %dx%d.', m, n, d, d);
    
    Mfd.dim = @() (m-1)*(n-1)*d*(d+1)/2;
    
    % Helpers to avoid computing full matrices simply to extract their trace
    vec     = @(A) A(:);
    trinner = @(A, B) real(vec(A')'*vec(B));  % = trace(A*B)
    trnorm  = @(A) sqrt((trinner(A, A))); % = sqrt(trace(A^2))
    

    function symmDblock = blocksymm(Dblock)
    	l = size(Dblock, 3);
    	symmDblock = Dblock;
    	for ll = 1:l
    		symmDblock(:,:,ll) = symm(Dblock(:,:,ll));
    	end 
    end

    % Riemannian metric as sum of the bi-linear metric on SPD matrices
    Mfd.inner = @innerproduct;
    function iproduct = innerproduct(X, eta, zeta) % BM: looks okay.
        iproduct = 0;
        for mm = 1 : m
            for nn = 1:n
                iproduct = iproduct + (trinner(X(:,:,mm,nn)\eta(:,:,mm,nn), X(:,:,mm,nn)\zeta(:,:,mm,nn)));
            end
        end
    end
    
    Mfd.norm = @innernorm;
    function inorm = innernorm(X, eta) % looks okay
        inorm = 0;
        for mm = 1:m
            for nn = 1:n
                inorm = inorm + (trnorm(X(:,:,mm,nn)\eta(:,:,mm,nn)))^2; 
            end
        end
        inorm = sqrt(inorm);
    end

    Mfd.typicaldist = @() sqrt(m*n*d*(d+1)/2); % BM: to be looked into.
    
    % Projection onto tangent space orthogonal to the metric
    Mfd.proj = @projection;

    function zeta = projection(X, eta)
        zeta = innerprojection(X, eta);
    end

    Mfd.orthproj = @innerprojection;
    function [zeta, Lambdasol, Thetasol] = innerprojection(X, eta)
        % RHSi = squeeze(- sum(eta, 4)); % d-by-d-by-m
        % RHSj = squeeze(- sum(eta, 3)); % d-by-d-by-n

        RHSi = reshape(- sum(eta, 4), [d d m]); % d-by-d-by-m
        RHSj = reshape(- sum(eta, 3), [d d n]); % d-by-d-by-n

        % Lambdasol is d-by-d-by-m
        % Thetasol is d-by-d-by-n
        [Lambdasol, Thetasol] = mylinearsolve(X, RHSi, RHSj);
        % [Lambdasol, Thetasol] = slowmylinearsolve(X, RHSi, RHSj);   

        zeta = zeros(size(eta));
        for mm = 1 : m
            for nn = 1 : n
                zeta(:,:,mm,nn) = eta(:,:,mm,nn) + (X(:,:,mm,nn)*(Lambdasol(:,:,mm)+Thetasol(:,:,nn))*X(:,:,mm,nn));
            end
        end
        
        %	% Debug
        %	neta = eta - zeta; % Normal vector
        %	innerproduct(X, zeta, neta) % This should be zero
    end

    function [Lambdasol, Thetasol] = slowmylinearsolve(X, RHSi, RHSj)
    	b = [RHSi(:); RHSj(:)];

    	XkronX = nan(d^2, d^2, m, n);
    	for mm = 1:m
    		for nn = 1:n
    			XkronX (:, :, mm, nn) = kron(X(:, :, mm,nn), X(:, :, mm,nn));
    		end
    	end
    	XkronXi = reshape(sum(XkronX, 4), [d^2 d^2 m]);
    	XkronXj = reshape(sum(XkronX, 3), [d^2 d^2 n]);

    	% Creating system matrix A
    	Amat = sparse((m+n)*d^2, (m+n)*d^2);

    	% off m by n blocks, each block of size d^2 by d^2
    	Amat = sparse((m+n)*d^2, (m+n)*d^2);
    	for mm = 1:m
    		for nn = 1: n
    			Amat(1+(mm-1)*d^2 : mm*d^2, m*d^2 + 1 + (nn-1)*d^2  :  m*d^2 + nn*d^2) = XkronX(:,:,mm,nn);
    		end
    	end
    	Amat = Amat + Amat';

    	% diag m by m blocks 
    	Amatrow = sparse(m*d^2, m*d^2);
    	for mm = 1 : m
    		Amatrow(1+(mm-1)*d^2 : mm*d^2, 1+(mm-1)*d^2 : mm*d^2) = XkronXi(:,:,mm);
    	end
    	Amat(1: m*d^2, 1 : m*d^2) = Amatrow;

    	% diag n by n blocks
    	Amatcol = sparse(n*d^2, n*d^2);
    	for nn = 1:n
    		Amatcol( 1+(nn-1)*d^2 : nn*d^2, 1+(nn-1)*d^2 : nn*d^2) = XkronXj(:,:,nn);
    	end
    	Amat(1 + m*d^2 : end,  1 + m*d^2 : end) = Amatcol;

    
    	% Backsolve for Ax = b.
    	vecsol = Amat \ b;

    	% Devectorize
    	lambdasol = vecsol(1 : m*d^2 , 1);
        thetasol = vecsol(1 + m*d^2  : end, 1);
        Lambdasol = blocksymm(reshape(lambdasol, [d  d m])); % block symm for numerical stability
        Thetasol = blocksymm(reshape(thetasol, [d d n])); % block symm for numerical stability


        % % debug
        % LHS = nan(size(X));
        % for mm = 1:m
        %     for nn = 1:n
        %         LHS(:,:,mm,nn) = X(:,:,mm,nn) * (  Lambdasol(:,:, mm)  +   Thetasol(:,:, nn) ) * X(:,:,mm,nn);
        %     end
        % end
        % LHSi = blocksymm(reshape(sum(LHS, 4), [d d m]));
        % LHSj = blocksymm(reshape(sum(LHS, 3), [d d n]));
        % lhs = [LHSi(:); LHSj(:)];
        % norm(lhs - b)/norm(b) % This should be zero
        % keyboard;

    end

    % solve the system of matrix linear equations
    function [Lambdasol, Thetasol] = mylinearsolve(X, RHSi, RHSj)
        % Solve the matrix linear system
        % sumj Xij (Lambdasoli + Thetasolj) Xij = RHSi and 
        % sumi Xij (Lambdasoli + Thetasolj) Xij = RHSj 
    
        % compute rhs of the matrix system
        rhs = [RHSi(:); RHSj(:)];

        % Call PCG
        % vecsol  = pcg(@compute_matrix_system, rhs, tolpcg, maxiterspcg);
        [vecsol, pcgflag, pcgrelres, pcgiter]  = pcg(@compute_matrix_system, rhs, tolpcg, maxiterspcg);

        % Devectorize vecsol into Lambdasol and Thetasol.
        lambdasol = vecsol(1 : m*d^2 , 1);
        thetasol = vecsol(1 + m*d^2  : end, 1);

        Lambdasol = blocksymm(reshape(lambdasol, [d  d m]));
        Thetasol = blocksymm(reshape(thetasol, [d  d n]));

        function lhs = compute_matrix_system(vecvar)
            veclambda = vecvar(1 :  m*d^2, 1); % length of size md^2
            vectheta = vecvar(1 +  m*d^2 : end, 1);    

            Lambda = blocksymm(reshape(veclambda, [d  d m]));
            Theta = blocksymm(reshape(vectheta, [d  d n]));

            LHS = nan(size(X));
            for mm = 1:m
                for nn = 1:n
                    LHS(:,:,mm,nn) = X(:,:,mm,nn) * (   Lambda(:,:, mm)  +   Theta(:,:, nn)  ) * X(:,:,mm,nn);
                end
            end

            % LHSi = squeeze(sum(LHS, 4));
            % LHSj = squeeze(sum(LHS, 3));

            LHSi = reshape(sum(LHS, 4), [d d m]);
            LHSj = reshape(sum(LHS, 3), [d d n]);

            lhs = [LHSi(:); LHSj(:)];
        end


		% % debug
		% LHS = nan(size(X));
		% for mm = 1:m
		% 	for nn = 1:n
		% 		LHS(:,:,mm,nn) = (X(:,:,mm,nn) * (Lambdasol(:,:, mm)  +   Thetasol(:,:, nn)) * X(:,:,mm,nn));
		% 	end
		% end
		% LHSi = reshape(sum(LHS, 4), [d d m]);
		% LHSj = reshape(sum(LHS, 3), [d d n]);

		% lhs = [LHSi(:); LHSj(:)];        
		% norm(lhs - rhs) % This should be zero
		% keyboard;

    end

    Mfd.tangent = Mfd.proj;
    Mfd.tangent2ambient = @(X, eta) eta;
    

    Mfd.egrad2rgrad = @egrad2rgrad; % AH: checked
    function rgrad = egrad2rgrad(X, egrad)
    	egradscaled = nan(size(egrad));
        for mm = 1:m
            for nn = 1:n
                egradscaled(:,:,mm,nn) = X(:,:,mm,nn)*symm(egrad(:,:,mm,nn))*X(:,:,mm,nn);
            end
        end
        rgrad = Mfd.proj(X, egradscaled);
        
        %	% Debug
        %   temp1 = sum(rgrad,3);
        %   temp2 = sum(rgrad,4);
        %   norm(temp1(:)) 
        %   norm(temp2(:)) 
        %   keyboard;
    end
    

    Mfd.ehess2rhess = @ehess2rhess;
    function Hess = ehess2rhess(X, egrad, ehess, eta)
        Hess = nan(size(X));
        egradscaled = nan(size(egrad));
        egradscaleddot = nan(size(egrad));
        for mm = 1:m
            for nn = 1:n
                egradmn = symm(egrad(:,:,mm,nn));
                ehessmn = symm(ehess(:,:,mm,nn));
                Xmn = X(:,:,mm,nn);
                etamn = eta(:,:,mm,nn);

                egradscaled(:,:,mm,nn) = Xmn*egradmn*Xmn;
                egradscaleddot(:,:,mm,nn) = Xmn*ehessmn*Xmn + 2*symm(etamn*egradmn*Xmn);
            end
        end
        
        % Compute Lambdasol and Thetasol
        % RHSj = squeeze(-sum(egradscaled,3)); % d-by-d-by-n
        % RHSi = squeeze(-sum(egradscaled,4)); % d-by-d-by-m

        RHSj = reshape(-sum(egradscaled,3), [d,d,n]); % d-by-d-by-n
        RHSi = reshape(-sum(egradscaled,4), [d,d,m]); % d-by-d-by-m

        [Lambdasol, Thetasol] = mylinearsolve(X, RHSi, RHSj);
        % [Lambdasol, Thetasol] = slowmylinearsolve(X, RHSi, RHSj);


        % Compute Lambdasoldot
        temp = nan(size(egrad));
        for mm = 1:m
            for nn = 1:n
                Xmn = X(:,:,mm,nn);
                etamn = eta(:,:,mm,nn);
                Lambdasoli = Lambdasol(:,:,mm);
                Thetasolj = Thetasol(:,:,nn);
                temp(:,:,mm,nn) = 2*symm(etamn*(Lambdasoli+Thetasolj)*Xmn);
            end
        end
        % RHSdotj = squeeze(-sum(egradscaleddot, 3) - sum(temp, 3));
        % RHSdoti = squeeze(-sum(egradscaleddot, 4) - sum(temp, 4));

        RHSdotj = reshape(-sum(egradscaleddot, 3) - sum(temp, 3), [d d n]);
        RHSdoti = reshape(-sum(egradscaleddot, 4) - sum(temp, 4), [d d m]);

        [Lambdasoldot, Thetasoldot] = mylinearsolve(X, RHSdoti, RHSdotj);


        for mm = 1:m
            for nn = 1:n
                egradmn = symm(egrad(:,:,mm,nn));
                ehessmn = symm(ehess(:,:,mm,nn));
                Xmn = X(:,:,mm,nn);
                etamn = eta(:,:,mm,nn);
                Lambdasoli = Lambdasol(:,:,mm);
                Thetasolj = Thetasol(:,:,nn);
                Lambdasoldoti = Lambdasoldot(:,:,mm);
                Thetasoldotj = Thetasoldot(:,:,nn);
                
                % Directional derivatives of the Riemannian gradient minus
                % the correction term
                rhessmn = (Xmn*(ehessmn + Lambdasoldoti + Thetasoldotj)*Xmn) + symm(etamn*(egradmn + Lambdasoli + Thetasolj)*Xmn);
                Hess(:,:,mm,nn) = rhessmn;
            end
        end
            
        Hess = Mfd.proj(X, Hess);
        
        %	% Debug
        %   temp1 = sum(Hess,3);
        %   temp2 = sum(Hess,4);
        %   norm(temp1(:)) 
        %   norm(temp2(:)) 
        %   keyboard;
        
    end
    
    Mfd.retr = @retraction; % AH: checked
    function Y = retraction(X, eta, t)
        if nargin < 3
            teta = eta;
        else
            teta = t*eta;
        end
     
        Y = nan(size(X));
        for mm=1:m
            for nn = 1:n
                % Second-order approximation of expm
                Y(:,:,mm,nn) = symm(X(:,:,mm,nn) + teta(:,:,mm,nn) + .5*teta(:,:,mm,nn)*((X(:,:,mm,nn) + myeps*eye(d) )\teta(:,:,mm,nn)));
                % expm
                %Y(:,:,mm,nn) = symm(X(:,:,mm,nn)*real(expm((X(:,:,mm,nn)  + myeps*eye(d))\(teta(:,:,mm,nn)))));
            end
        end
        Y = real(blocksinkhornwithSPDblocks(Y, M, N, maxitersblockDS, checkperiodblockDS, tolblockDS));
        %	% Debug
        %   temp1 = reshape(sum(Y,3), [d d n]) - N;
        %   temp2 = reshape(sum(Y,4), [d d m]) - M;
        %   norm(temp1(:));
        %   norm(temp2(:));
        %   keyboard;
    end
    
    Mfd.hash = @(X) ['z' hashmd5(X(:))];
    
    Mfd.rand = @random;
    function X = random()
        X = nan(d,d,m,n);
        for mm = 1:m
            for nn = 1:n
                D = diag(1+rand(d, 1));
                [Q, R] = qr(randn(d)); 
                X(:,:,mm,nn) = Q*D*Q';
            end
        end
        X = blocksinkhornwithSPDblocks(X, M, N, maxitersblockDS, checkperiodblockDS, tolblockDS);
    end

    % Generate a uniformly random unit-norm tangent vector at X.
    Mfd.randvec = @randomvec;
    function eta = randomvec(X)
        eta = nan(size(X));
        for mm = 1:m
            for nn = 1:n
                eta(:,:,mm,nn) = symm(randn(d));
            end
        end
        eta = Mfd.proj(X, eta);
        nrm = Mfd.norm(X, eta);
        eta = eta ./ nrm;
    end
    
    Mfd.lincomb = @matrixlincomb;
    
    Mfd.zerovec = @(X) zeros(d, d, m, n);
    
    % Poor man's vector transport: exploit the fact that all tangent spaces
    % are the set of symmetric matrices, so that the identity is a sort of
    % vector transport. It may perform poorly if the origin and target (X1
    % and X2) are far apart though. This should not be the case for typical
    % optimization algorithms, which perform small steps.
    Mfd.transp = @(X1, X2, eta) Mfd.proj(X2, eta);
    
    % vec and mat are not isometries, because of the unusual inner metric.
    Mfd.vec = @(X, U) U(:);
    Mfd.mat = @(X, u) reshape(u, d, d, m, n);
    Mfd.vecmatareisometries = @() false;
end


