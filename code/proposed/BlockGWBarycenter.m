function [Dbar] = BlockGWBarycenter(D1, D2, w, d, X1, X2, Xbar, options)

    % D1, D2 are the distance matrix of size n-by-n
    
    % w is the weight should be [t, 1-t]
    % X1, X2, Xbar are the marginals of size (d,d,n)

    % Dbar is the barycenter distance matrix of size n-by-n
    
    n = size(D1,1);
    
    myI = eye(d);
    symm = @(X) (X + X')/2;

    localdefaults.lambda = 0;
    localdefaults.tolblockDS = 1e-9;
    localdefaults.tolgradnorm = 1e-6;
    localdefaults.GWOTiter = 50; % inner iteration
    localdefaults.BWBaryiter = 30; % outter iteration
    localdefaults.tolpcg = 1e-6;
    localdefaults.init = [];
    localdefaults.method = 'SD';
    options = mergeOptions(localdefaults, options);

    tolblockDS = options.tolblockDS;
    tolgradnorm = options.tolgradnorm;
    tolpcg = options.tolpcg;
    lambda = options.lambda;
    
    
    % compute trace(Xbar_i Xbar_j)
    % make a scalar times identity matrix used later
    trXbarij = nan(n, n);
    trXbar = nan(d,d,n,n);
    D1block = zeros(d,d,n,n);
    D2block = zeros(d,d,n,n);
    for ii = 1 : n
        for jj = ii : n
            temp = trace(Xbar(:,:,ii) * Xbar(:,:,jj));
            trXbarij(ii,jj) = temp;
            trXbarij(jj,ii) = temp;
            trXbar(:,:,ii,jj) = (1/temp) * eye(d);
            trXbar(:,:,jj,ii) = (1/temp) * eye(d);
            
            temp = D1(ii,jj) * myI;
            D1block(:,:,ii,jj) = temp;
            D1block(:,:,jj,ii) = temp;
            
            temp = D2(ii,jj) * myI;
            D2block(:,:,ii,jj) = temp;
            D2block(:,:,jj,ii) = temp;
        end
    end
    D1block = reshape(permute(D1block, [1 3 2 4]), [d*n, d*n]);
    D2block = reshape(permute(D2block, [1 3 2 4]), [d*n, d*n]);
    Dblocks{1} = D1block;
    Dblocks{2} = D2block;
    
    
    Dbar = w(1) * D1 + w(2) * D2; %init
    Ds{1} = D1;
    Ds{2} = D2;
    Xs{1} = X1;
    Xs{2} = X2;
    
    myinit = repmat(eye(d), [1 1 n n]);
    myinit = blocksinkhornwithSPDblocks(myinit, X1, X2, 1000,[],1e-6);
    Gamma{1} = myinit;
    Gamma{2} = myinit;
    for iter = 1 : options.BWBaryiter
        
        obj = 0;
        % update Gamma
        for ell = 1 : 2
            gwot_options.tolblockDS = tolblockDS;
            gwot_options.tolgradnorm = tolgradnorm;
            gwot_options.tolpcg = tolpcg;
            gwot_options.lambda = lambda;
            GWOTiter = options.GWOTiter;
            gwot_options.maxiter = floor(GWOTiter(iter));
            
            gwot_options.method = options.method;
            gwot_options.init = Gamma{ell};
            [Gammaell, obj_ell] = compute_gw_coupling(Ds{ell}, Dbar, d, Xs{ell}, Xbar, gwot_options);
            
            obj = obj + w(ell) * obj_ell;
            Gamma{ell} = Gammaell;
            
            % convert gamma into block form
            Gammablock{ell} = reshape(permute(Gammaell, [1 3 2 4]), [d*n, d*n]);
        end        
        
        % update Dbar
        blockDbar = 0;
        for ell = 1 : 2
            blockDbar = blockDbar + w(ell) * Gammablock{ell}' * Dblocks{ell} * Gammablock{ell};
        end
                
        
        Dbarnew = zeros(n,n);
        for ii = 1 : n
            for jj = (ii + 1) : n
                blockDbar_ij = blockDbar( (ii-1)*d+1 : ii*d, (jj-1)*d+1 : jj*d);
                temp = trace(blockDbar_ij)/trXbarij(ii,jj);
                Dbarnew(ii,jj) = temp;
                Dbarnew(jj,ii) = temp;
            end
        end
        %}
        
        diff = Dbar - Dbarnew;
        diff = norm(diff(:))/norm(Dbar(:));
        
        fprintf('\n ==== Iteration %d Finished! ==== \n',iter);
        fprintf('Difference: %.2f, \t obj: %.2f\n', diff, obj);
        fprintf('\n');
        
        Dbar = Dbarnew;
        
        if diff <= 1e-4
            break;
        end
    end
    

end

